/*
 * world_builder C++ supervisor controller.
 */
#include <webots/Supervisor.hpp>
#include <webots/Emitter.hpp>
#include <webots/Receiver.hpp>
#include <iostream>
#include <string>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>



#include <QFile>
#include <QXmlSimpleReader>
#include <QXmlInputSource>

#include "cblabhandler.h"

// Define the default maximum duration of the simulation in seconds
// This value can be overridden by the envoronment variable MAX_TIME
// Example: export MAX_TIME=60
double MAX_TIME_SECONDS = 300.0;

#define M_PI 3.14159265358979323846

struct cell_t
{
    int x, y;
} controlCellPath[1024], newCell;

int nCellPath = 0;
int nextPathInd = 0;
int scoreControl = 0;

// Get the e-puck node by DEF name (e.g., "EPUCK")
webots::Node *epuck_node;

// --- GLOBAIS ADICIONADAS PARA O RESET ---
webots::Emitter *emitter;
webots::Field *trans_field; 
double start_pos[3] = {0, 0, 0};
webots::Receiver *receiver_sup;
// ---------------------------------------

#define PATHCUBESIZE (0.15)

pid_t get_sibling_pid() {
    FILE *fp;
    char buffer[1024];
    char command[1024];
    pid_t sibling_pid=0;

    // Construct the command string:
    // pgrep -P [my_parent_pid] | grep -v [my_pid]
    snprintf(command, sizeof(command), 
             "pgrep -P %d | grep -v %d", getppid(), getpid());

    // Open a pipe to execute the command
    fp = popen(command, "r");
    if (fp == NULL) {
        perror("popen failed");
        return 0;
    }

    // Read the output line by line (each line contains one PID)
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        // Convert the string PID to an integer
        sibling_pid = atoi(buffer);
    }

    // Close the pipe
    pclose(fp);
    
    return sibling_pid;
}

bool is_pid_running(int pid) {
    if (pid <= 0) return false;
    // Sending signal 0 to a PID checks if the process is alive without killing it.
    // Returns 0 if process exists, -1 if it doesn't.
    return (kill(pid, 0) == 0);
}

struct cell_t getRobotCell()
{
    struct cell_t cell;
    if (epuck_node)
    {
        const double *position = epuck_node->getPosition();
        // std::cout << "e-puck position: x=" << position[0] << " y=" << position[1] << " z=" << position[2] << std::endl;
        cell.x = static_cast<int>(position[0] / PATHCUBESIZE);
        cell.y = static_cast<int>(position[1] / PATHCUBESIZE);
    }
    return cell;
}

void build_cell_path(cbLab *lab)
{
    controlCellPath[0].x = lab->Target(0)->Center().x / PATHCUBESIZE;
    controlCellPath[0].y = lab->Target(0)->Center().y / PATHCUBESIZE;

    fprintf(stderr, "::: %d, %d %d %f\n", controlCellPath[0].x, controlCellPath[0].y, lab->nTargets(), lab->Target(0)->Center().x);

    //controlCellPath[0].x = 1;
    //controlCellPath[0].y = 5;
    struct cell_t newCell = controlCellPath[0];
    newCell.x++;
    nCellPath = 1;
    int dir = 0;
    while (newCell.x != controlCellPath[0].x || newCell.y != controlCellPath[0].y)
    {
        int d;
        struct cell_t cell = newCell;
        controlCellPath[nCellPath] = cell;
        nCellPath++;
        int test_dirs[3] = {0, -90, 90};
        for (d = 0; d < 3; d++)
        {
            if (lab->reachable(cbPoint(newCell.x * PATHCUBESIZE + PATHCUBESIZE / 2.0, newCell.y * PATHCUBESIZE + PATHCUBESIZE / 2.0),
                               cbPoint(newCell.x * PATHCUBESIZE + PATHCUBESIZE / 2.0 + cos((test_dirs[d] + dir) * M_PI / 180.0) * PATHCUBESIZE, newCell.y * PATHCUBESIZE + PATHCUBESIZE / 2.0 + sin((test_dirs[d] + dir) * M_PI / 180.0) * PATHCUBESIZE)))
            {
                newCell.x = round(cell.x + cos((dir + test_dirs[d]) * M_PI / 180.0));
                newCell.y = round(cell.y + sin((dir + test_dirs[d]) * M_PI / 180.0));
                dir = (dir + test_dirs[d] + 360) % 360;

                break;
            }
        }
        if (d == 3)
        {
            fprintf(stderr, "Lab has no Loop! Does not fit for this challenge!\n"); // TODO: add Graphical Window Warning

            exit(1);
        }
    }

    // debug
    //  for(int i=0; i<nCellPath; i++) {
    //        printf("pathcell %d %d\n",controlCellPath[i].x, controlCellPath[i].y);
    //  }
}

void update_score()
{   
    struct cell_t curCell = getRobotCell();
    if (curCell.x == controlCellPath[nextPathInd].x && curCell.y == controlCellPath[nextPathInd].y)
    {   
        nextPathInd++;
        if (nextPathInd >= nCellPath)
            nextPathInd = 0;
        scoreControl += 10;
        if (emitter) {
            emitter->send(&scoreControl, sizeof(int));
        }
    }
}

int main(int argc, char **argv)
{
    // ---
    // 1. INITIALIZATION
    // ---
    webots::Supervisor *supervisor = new webots::Supervisor();
    int timeStep = (int)supervisor->getBasicTimeStep();

    // --- CONFIGURAÇÃO EMITTER / RECEIVER ---
    emitter = supervisor->getEmitter("emitter_supervisor");
    if(emitter) emitter->setChannel(1);

    receiver_sup = supervisor->getReceiver("receiver_supervisor");
    if (receiver_sup) {
        receiver_sup->enable(timeStep);
        receiver_sup->setChannel(2); 
    }

    if (emitter == NULL) std::cerr << "ERRO: Emitter do Supervisor não encontrado!" << std::endl;
    if (receiver_sup == NULL) std::cerr << "ERRO: Receiver do Supervisor não encontrado!" << std::endl;
    // ---------------------------------------

    // ---
    // 1a. GET MAXIMUM SIMULATION TIME
    // ---

    char *maxTimeEnv = getenv("MAX_TIME");
    if (maxTimeEnv != NULL)
    {
        char *pend;
        double maxTime = strtod(maxTimeEnv, &pend);
        if(*pend=='\0' && pend != maxTimeEnv) {
             MAX_TIME_SECONDS = maxTime;
        }
        else {
            fprintf(stderr,"Could not get max_time from MAX_TIME value (%s)\n", maxTimeEnv);
        }
    }
    std::cerr << "Maximum Simulation Time: " << MAX_TIME_SECONDS << " s\n";

    // ---
    // 2. GET SCENE TREE NODES
    // ---

    // Get the root node of the scene tree
    webots::Node *root_node = supervisor->getRoot();
    // Get the 'children' field of the root node, where we'll add new objects.
    webots::Field *children_field = root_node->getField("children");

    // ---
    // 3. DEFINE AND CREATE NEW OBJECTS
    // ---

    // Read lab file and create maze walls in webots
    char lab_filename[1024 * 8] = "C2-lab.xml";
    QXmlInputSource *source;

    QFile srcFile(lab_filename);

    if (!srcFile.exists())
    {
        std::cerr << "Could not open " << lab_filename << "\n";
        exit(0);
    }
    if ((source = new QXmlInputSource(&srcFile)) == 0)
    {
        std::cerr << "Fail sourcing lab file\n";
        exit(0);
    }

    cbLabHandler *labHandler = new cbLabHandler;
    labHandler->setChildrenField(children_field);

    QXmlSimpleReader xmlParser;
    xmlParser.setContentHandler(labHandler);

    xmlParser.setErrorHandler(labHandler);

    if (!xmlParser.parse(source))
    {
        std::cerr << "Error parsing lab file\n";
        exit(0);
    }

    // create the cell path of the closed loop to be used by update_score
    build_cell_path(labHandler->getLab());

    // --- ROBOT SETUP CORRIGIDO ---
    // get robot
    epuck_node = supervisor->getFromDef("EPUCK");
    
    // Obter o campo 'translation' DEPOIS de encontrar o robô (isto evita o crash)
    trans_field = epuck_node->getField("translation");

    // Reposition robot to first target area of maze
    webots::Field *translationField = epuck_node->getField("translation");
    if (translationField) {
        // Define the new position (e.g., move to X=1.0, Y=2.0)
        double newTranslation[3] = {labHandler->getLab()->Target(0)->Center().x, 
                                    labHandler->getLab()->Target(0)->Center().y, 
                                    0.0}; 
        translationField->setSFVec3f(newTranslation);
        //std::cout << "E-puck repositioned." << std::endl;
    }
    
    // Guardar a posição inicial para o RESET
    const double *pos = trans_field->getSFVec3f();
    start_pos[0] = pos[0];
    start_pos[1] = pos[1];
    start_pos[2] = pos[2];
    // ----------------------------
    
    // Get the PID of robot controller
    int robot_pid = get_sibling_pid();

    if (robot_pid == 0) {
        fprintf(stderr,"Could not get Robot Controller PID.\n");
        supervisor->simulationQuit(2);
    }

    webots::Field *rot_field = epuck_node->getField("rotation");
    double start_rot[4] = {0,0,1,0};
    if (rot_field) {
    const double *rot = rot_field->getSFRotation();
    start_rot[0] = rot[0];
    start_rot[1] = rot[1];
    start_rot[2] = rot[2];
    start_rot[3] = rot[3];
    }

    // ---
    // 4. MAIN LOOP (Optional)
    // ---

    // The main work is done, but the controller must keep running
    // for the simulation to continue.
    while (supervisor->step(timeStep) != -1)
    {
        // --- CÓDIGO DE RESET ADICIONADO ---
        while (receiver_sup && receiver_sup->getQueueLength() > 0) {
            const char *msg = (const char *)receiver_sup->getData();
            int dataSize = receiver_sup->getDataSize();

            if (dataSize >= 5 && strncmp(msg, "RESET", 5) == 0) {
                static double lastResetTime = -1.0;
                if (supervisor->getTime() > lastResetTime + 0.5) { 
                    std::cout << ">>> SUPERVISOR: Executando RESET e limpando fila... <<<" << std::endl;
                    
                    trans_field->setSFVec3f(start_pos);
                    webots::Field *rot_field = epuck_node->getField("rotation");
                    double start_rot[4] = {0, 0, 1, 0};
                    if (rot_field) rot_field->setSFRotation(start_rot);
                    
                    epuck_node->resetPhysics();
                    
                    scoreControl = 0;
                    nextPathInd = 0;
                    
                    // int zero = 0;
                    // emitter->send(&zero, sizeof(int));
                    lastResetTime = supervisor->getTime();
                }
            }
            receiver_sup->nextPacket(); 
        }
        // ----------------------------------

        // Get the current simulation time.
        double currentTime = supervisor->getTime();

        if (robot_pid > 0 && !is_pid_running(robot_pid)) {
            printf("Robot controller finished at time %f.\n", currentTime);
            
            //supervisor->simulationQuit(0); 
            break;
        }

        std::string scoreText;

        // Check if the simulation time has exceeded the maximum duration.
        
        // if (currentTime >= MAX_TIME_SECONDS)
        // {
        //     // The argument 0 indicates a successful exit.
        //     scoreText = "Final Score: " + std::to_string(scoreControl);
        // }
        // else {
        //     // Increment the score (this is just an example, replace with your logic)

        // }
        update_score();
        scoreText = "Score: " + std::to_string(scoreControl);
        // Display the label
        supervisor->setLabel(0, scoreText, 0.6, 0.01, 0.1, 0xFF0000, 0.0, "Arial");
    }

    delete supervisor;
    return 0;
}