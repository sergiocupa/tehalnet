#include "../include/tehalnet.h"
#include "../../tehalhost/include/tehalhost.h"
#include "tehalnet_plug.h"
#include "yason.h"
#include "eventhub.h"
#include <windows.h>
#include <string.h>
#include <stdio.h>


bool Initialized  = false;



bool teh_init(bool use_device, int device_index, char* devide_lib)
{
	if (Initialized) return true;

	teh_plug_assemble(use_device, device_index, devide_lib);

	Initialized  = true;
}


