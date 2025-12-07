import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:05:38.460637
# Source Brief: brief_02093.md
# Brief Index: 2093
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A real-time strategy Gymnasium environment where the agent manages a swarm of 
    resource-gathering drones. The goal is to achieve a balanced resource 
    distribution (20% of 5 different resource types) across all drone inventory
    slots before running out of space or time.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Manage a swarm of drones to collect different resources. Achieve a perfectly balanced inventory across all drones to win."
    user_guide = "↑↓: Select drone. ←→: Select resource type. Space: Assign task. Shift: Recall all drones."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    NUM_DRONES = 5
    DRONE_INVENTORY_SIZE = 4
    NUM_RESOURCES = 5
    DRONE_SPEED = 2.5
    COLLECTION_RATE = 0.05  # Units per step
    COLLECTION_RADIUS = 20
    NODE_INITIAL_AMOUNT = 10.0
    MAX_STEPS = 1500

    # --- Visuals ---
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 50)
    COLOR_DRONE = (220, 220, 240)
    COLOR_DRONE_SELECTED_GLOW = (255, 255, 100)
    RESOURCE_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    COLOR_UI_TEXT = (200, 200, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 14)
        self.font_medium = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        # Game state variables to be initialized in reset()
        self.drones = []
        self.resource_nodes = []
        self.particles = []
        self.selected_drone_idx = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_action = np.array([0, 0, 0])
        
        # Use np.random.Generator for modern Numpy API
        self.np_random = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize the random number generator
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_drone_idx = 0
        self.prev_action = np.array([0, 0, 0])
        self.particles.clear()
        
        # Initialize Drones
        self.drones = []
        for i in range(self.NUM_DRONES):
            angle = 2 * math.pi * i / self.NUM_DRONES - math.pi / 2
            x = self.SCREEN_WIDTH / 2 + math.cos(angle) * 60
            y = self.SCREEN_HEIGHT / 2 + math.sin(angle) * 60
            self.drones.append({
                "pos": pygame.Vector2(x, y),
                "target_node_idx": None,
                "inventory": [],
                "collection_progress": 0.0,
                "state": "idle", # idle, moving_to_resource, collecting
                "staged_target_id": i % self.NUM_RESOURCES,
            })

        # Initialize Resource Nodes
        self.resource_nodes = []
        for i in range(self.NUM_RESOURCES):
            angle = 2 * math.pi * i / self.NUM_RESOURCES - math.pi / 2
            radius = self.SCREEN_HEIGHT / 2 - 50
            x = self.SCREEN_WIDTH / 2 + math.cos(angle) * radius
            y = self.SCREEN_HEIGHT / 2 + math.sin(angle) * radius
            self.resource_nodes.append({
                "pos": pygame.Vector2(x, y),
                "resource_id": i,
                "amount": self.NODE_INITIAL_AMOUNT,
            })
            
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01 # Small penalty for each step to encourage efficiency
        
        self._handle_input(action)
        reward += self._update_game_state()
        self._update_particles()
        
        terminated, win = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated:
            if win:
                reward += 100.0  # Win bonus
            else:
                reward -= 100.0  # Loss penalty
        
        self.game_over = terminated or truncated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        # Unpack factorized action
        movement_action = action[0]
        space_press = action[1] == 1 and self.prev_action[1] == 0
        shift_press = action[2] == 1 and self.prev_action[2] == 0
        
        # Only register movement on change to simulate a key press
        movement = 0
        if movement_action != 0 and movement_action != self.prev_action[0]:
            movement = movement_action
        
        self.prev_action = action.copy()

        # Cycle selected drone
        if movement == 1: self.selected_drone_idx = (self.selected_drone_idx - 1 + self.NUM_DRONES) % self.NUM_DRONES
        elif movement == 2: self.selected_drone_idx = (self.selected_drone_idx + 1) % self.NUM_DRONES
            
        # Cycle staged resource for the selected drone
        selected_drone = self.drones[self.selected_drone_idx]
        if movement == 3: selected_drone["staged_target_id"] = (selected_drone["staged_target_id"] - 1 + self.NUM_RESOURCES) % self.NUM_RESOURCES
        elif movement == 4: selected_drone["staged_target_id"] = (selected_drone["staged_target_id"] + 1) % self.NUM_RESOURCES
            
        # Assign task to drone
        if space_press:
            staged_id = selected_drone["staged_target_id"]
            node_idx = next((i for i, node in enumerate(self.resource_nodes) if node["resource_id"] == staged_id), None)
            if node_idx is not None and len(selected_drone["inventory"]) < self.DRONE_INVENTORY_SIZE:
                selected_drone["target_node_idx"] = node_idx
                selected_drone["state"] = "moving_to_resource"
        
        # Recall all drones
        if shift_press:
            for drone in self.drones:
                drone["target_node_idx"] = None
                drone["state"] = "idle"

    def _update_game_state(self):
        step_reward = 0
        for drone in self.drones:
            if drone["state"] == "moving_to_resource":
                if drone["target_node_idx"] is None:
                    drone["state"] = "idle"
                    continue
                
                target_node = self.resource_nodes[drone["target_node_idx"]]
                if target_node["amount"] <= 0:
                    drone["target_node_idx"], drone["state"] = None, "idle"
                    continue

                direction = target_node["pos"] - drone["pos"]
                if direction.length() < self.COLLECTION_RADIUS:
                    drone["state"] = "collecting"
                else:
                    drone["pos"] += direction.normalize() * self.DRONE_SPEED
            
            elif drone["state"] == "collecting":
                if drone["target_node_idx"] is None:
                    drone["state"] = "idle"
                    continue
                
                target_node = self.resource_nodes[drone["target_node_idx"]]
                if target_node["amount"] <= 0 or len(drone["inventory"]) >= self.DRONE_INVENTORY_SIZE:
                    drone["target_node_idx"], drone["state"] = None, "idle"
                    continue
                
                collected_amount = min(self.COLLECTION_RATE, target_node["amount"])
                target_node["amount"] -= collected_amount
                drone["collection_progress"] += collected_amount
                
                if drone["collection_progress"] >= 1.0:
                    drone["collection_progress"] -= 1.0
                    if len(drone["inventory"]) < self.DRONE_INVENTORY_SIZE:
                        res_id = target_node["resource_id"]
                        drone["inventory"].append(res_id)
                        
                        counts = self._get_total_resource_counts()
                        target_per_res = (self.NUM_DRONES * self.DRONE_INVENTORY_SIZE) / self.NUM_RESOURCES
                        
                        if counts[res_id] <= target_per_res:
                            step_reward += 1.0  # Correct resource collected
                        else:
                            step_reward -= 1.0  # Excess resource collected
                        
                        self._create_particles(drone["pos"], self.RESOURCE_COLORS[res_id], 3)
        return step_reward

    def _check_termination(self):
        total_filled_slots = sum(len(d["inventory"]) for d in self.drones)
        total_possible_slots = self.NUM_DRONES * self.DRONE_INVENTORY_SIZE
        
        if total_filled_slots >= total_possible_slots:
            counts = self._get_total_resource_counts()
            target_per_res = total_possible_slots / self.NUM_RESOURCES
            is_win = all(count == target_per_res for count in counts.values())
            return True, is_win
        
        if self.steps >= self.MAX_STEPS:
            # This condition is now for truncation, not termination
            return False, False
            
        return False, False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_game()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Render Resource Nodes
        for node in self.resource_nodes:
            radius = int(5 + (node["amount"] / self.NODE_INITIAL_AMOUNT) * 15)
            if radius > 0:
                color = self.RESOURCE_COLORS[node["resource_id"]]
                pos_int = (int(node["pos"].x), int(node["pos"].y))
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, color)

        # Render Drones
        for i, drone in enumerate(self.drones):
            pos_int = (int(drone["pos"].x), int(drone["pos"].y))
            
            if i == self.selected_drone_idx:
                # Selection Glow
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 15, self.COLOR_DRONE_SELECTED_GLOW + (50,))
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 15, self.COLOR_DRONE_SELECTED_GLOW + (100,))
                
                # Staged Target Line
                staged_node_idx = next((idx for idx, node in enumerate(self.resource_nodes) if node["resource_id"] == drone["staged_target_id"]), None)
                if staged_node_idx is not None:
                    target_pos = self.resource_nodes[staged_node_idx]["pos"]
                    color = self.RESOURCE_COLORS[drone["staged_target_id"]]
                    pygame.draw.aaline(self.screen, color, pos_int, (int(target_pos.x), int(target_pos.y)), blend=1)
            
            # Task outline
            if drone["target_node_idx"] is not None:
                res_id = self.resource_nodes[drone["target_node_idx"]]["resource_id"]
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 10, self.RESOURCE_COLORS[res_id])

            # Drone body
            size = 8
            points = [(pos_int[0], pos_int[1] - size), (pos_int[0] - size/1.5, pos_int[1] + size/2), (pos_int[0] + size/1.5, pos_int[1] + size/2)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_DRONE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_DRONE)

            # Inventory UI
            inv_size = 6
            start_x = pos_int[0] - (self.DRONE_INVENTORY_SIZE * (inv_size + 1)) / 2
            start_y = pos_int[1] - 25
            for j in range(self.DRONE_INVENTORY_SIZE):
                rect = pygame.Rect(start_x + j * (inv_size + 1), start_y, inv_size, inv_size)
                if j < len(drone["inventory"]):
                    pygame.draw.rect(self.screen, self.RESOURCE_COLORS[drone["inventory"][j]], rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

    def _render_ui(self):
        pygame.draw.rect(self.screen, (0,0,0,150), (0, 0, self.SCREEN_WIDTH, 35))

        score_text = self.font_medium.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 8))
        
        steps_text = self.font_medium.render(f"Step: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 8))

        # Overall Resource Distribution Bar
        counts = self._get_total_resource_counts()
        total_slots = self.NUM_DRONES * self.DRONE_INVENTORY_SIZE
        bar_width, bar_height, bar_x, bar_y = 300, 12, (self.SCREEN_WIDTH - 300) / 2, 10
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x -1, bar_y -1, bar_width + 2, bar_height + 2), 1)
        current_x = bar_x
        for i in range(self.NUM_RESOURCES):
            width = (counts.get(i, 0) / max(1, total_slots)) * bar_width
            pygame.draw.rect(self.screen, self.RESOURCE_COLORS[i], (current_x, bar_y, width, bar_height))
            current_x += width
        
        for i in range(1, self.NUM_RESOURCES):
            marker_x = bar_x + (i / self.NUM_RESOURCES) * bar_width
            pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (marker_x, bar_y), (marker_x, bar_y + bar_height))

        if self.game_over:
            terminated, is_win = self._check_termination()
            if not terminated: # check if game over was due to truncation
                 is_win = False
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            end_text_str = "SUCCESS" if is_win else "MISSION FAILED"
            end_color = (100, 255, 100) if is_win else (255, 100, 100)
            end_text = self.font_large.render(end_text_str, True, end_color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2)))

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)).normalize() * self.np_random.uniform(0.5, 2.0),
                "lifespan": self.np_random.integers(15, 31),
                "color": color
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30.0))
            size = max(1, int(3 * (p["lifespan"] / 30.0)))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, p["color"] + (alpha,), (size, size), size)
            self.screen.blit(temp_surf, (int(p["pos"].x) - size, int(p["pos"].y) - size), special_flags=pygame.BLEND_RGBA_ADD)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "resource_counts": self._get_total_resource_counts(),
            "selected_drone": self.selected_drone_idx
        }
        
    def _get_total_resource_counts(self):
        counts = {i: 0 for i in range(self.NUM_RESOURCES)}
        for drone in self.drones:
            for res_id in drone["inventory"]:
                counts[res_id] += 1
        return counts

    def close(self):
        pygame.quit()
        
# --- Main block for human play testing ---
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Un-dummy the video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Drone Resource Management")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    action = np.array([0, 0, 0])
    
    print("\n--- Human Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset Environment")
    print("----------------------\n")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0.0
            
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]: action[0] = 0
                elif event.key == pygame.K_SPACE: action[1] = 0
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']:.2f}")
            # Render final frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(3000)
            
            obs, info = env.reset()
            total_reward = 0.0

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    env.close()