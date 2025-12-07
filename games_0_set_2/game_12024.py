import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:54:02.269294
# Source Brief: brief_02024.md
# Brief Index: 2024
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Repair a microscopic cell from within by building a scaffolding network for your nanobots. "
        "Manage your resources and use a time-warp ability to hold back the encroaching damage."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to build scaffolding nodes. "
        "Press shift to toggle the time-slowing 'chrono-warp' ability."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # Game logic runs at this speed
        self.MAX_STEPS = 600 * self.FPS # 10 minutes

        # Colors
        self.COLOR_BG = (15, 20, 45)
        self.COLOR_CELL_WALL = (40, 50, 90)
        self.COLOR_CELL_WALL_OUTLINE = (80, 100, 180)
        self.COLOR_DAMAGE = (255, 50, 50)
        self.COLOR_REPAIR = (50, 255, 100)
        self.COLOR_NANO = (50, 200, 255)
        self.COLOR_NANO_GLOW = (150, 230, 255)
        self.COLOR_SCAFFOLD = (150, 150, 150)
        self.COLOR_SCAFFOLD_GLOW = (200, 200, 200)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_CHRONO_BAR = (0, 150, 255)
        self.COLOR_INTEGRITY_BAR = (50, 255, 100)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Game state variables (to be initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.cell_integrity = None
        self.time_warp_active = None
        self.chrono_energy = None
        self.shift_was_held = None
        self.space_was_held = None
        self.cursor_pos = None
        self.scaffolding_nodes = None
        self.nanobots = None
        self.damage_sites = None
        self.damage_spawn_points = None
        self.time_to_next_damage = None
        self.damage_spawn_cooldown_base = None
        self.repairs_completed = None
        self.nanobot_repair_speed_multiplier = None
        
        # Initialize state variables
        # self.reset() is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cell_integrity = 100.0

        # Time manipulation state
        self.time_warp_active = False
        self.chrono_energy = 100.0
        self.shift_was_held = False
        self.space_was_held = False

        # Player cursor
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        
        # Scaffolding
        self.scaffold_grid_size = 40
        self.scaffolding_nodes = {}
        hub_pos = self._get_grid_pos(np.array([self.WIDTH / 2, self.HEIGHT / 2]))
        self.scaffolding_nodes[hub_pos] = set()

        # Nanobots
        self.nanobots = [self._create_nanobot() for _ in range(3)]
        
        # Damage
        self.damage_sites = []
        self._define_damage_spawn_points()
        self.damage_spawn_cooldown_base = 5 * self.FPS
        self.time_to_next_damage = self.damage_spawn_cooldown_base

        # Progression
        self.repairs_completed = 0
        self.nanobot_repair_speed_multiplier = 1.0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        # Determine game speed from time warp
        self._update_time_warp(shift_held)
        game_speed = 0.5 if self.time_warp_active and self.chrono_energy > 0 else 1.0

        # Handle player input
        self._handle_cursor_movement(movement)
        place_scaffold_reward = self._handle_scaffolding_placement(space_held)
        reward += place_scaffold_reward

        # Update game logic
        integrity_lost = self._update_damage(game_speed)
        repair_reward, new_repairs = self._update_nanobots(game_speed)
        
        # Calculate rewards
        reward += repair_reward
        if new_repairs > 0:
            reward += 5 * new_repairs
            self.repairs_completed += new_repairs
            # Check for upgrades
            if self.repairs_completed > 0 and self.repairs_completed % 20 == 0:
                self.nanobot_repair_speed_multiplier *= 1.10
                # Could also unlock new scaffold types here if implemented
        
        # Reward for integrity change (penalize loss, reward gain)
        # Note: integrity_lost is positive for a loss
        # The brief says "+0.1 for each percentage point of cell integrity repaired"
        # We can calculate this by seeing how much integrity *would have been* lost vs how much was actually lost
        self.initial_damage_per_step = 0.01 # Damage per site per step
        integrity_repaired_this_step = (self.initial_damage_per_step * len(self.damage_sites) * game_speed) - integrity_lost
        reward += integrity_repaired_this_step * 0.1

        self.score += reward

        # Update action state for next step
        self.space_was_held = space_held
        self.shift_was_held = shift_held
        
        # Check termination conditions
        terminated = self._check_termination()
        if terminated:
            reward -= 100
            self.score -= 100
            self.game_over = True
        
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Update Sub-routines ---

    def _update_time_warp(self, shift_held):
        # Toggle time warp on button press (rising edge)
        if shift_held and not self.shift_was_held:
            self.time_warp_active = not self.time_warp_active
            # sfx: time_warp_on.wav / time_warp_off.wav

        if self.time_warp_active and self.chrono_energy > 0:
            self.chrono_energy -= 0.5 # Depletion rate
            if self.chrono_energy <= 0:
                self.time_warp_active = False # Auto-disable
                # sfx: time_warp_fizzle.wav
        else:
            self.chrono_energy = min(100.0, self.chrono_energy + 0.25) # Recharge rate

    def _handle_cursor_movement(self, movement):
        cursor_speed = 10.0
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

    def _handle_scaffolding_placement(self, space_held):
        reward = 0
        if space_held and not self.space_was_held:
            grid_pos = self._get_grid_pos(self.cursor_pos)
            if grid_pos not in self.scaffolding_nodes:
                self.scaffolding_nodes[grid_pos] = set()
                # Connect to adjacent nodes
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    neighbor_pos = (grid_pos[0] + dx, grid_pos[1] + dy)
                    if neighbor_pos in self.scaffolding_nodes:
                        self.scaffolding_nodes[grid_pos].add(neighbor_pos)
                        self.scaffolding_nodes[neighbor_pos].add(grid_pos)
                reward = 1.0 # Reward for placing a new piece
                # sfx: scaffold_place.wav
        return reward

    def _update_damage(self, game_speed):
        # Spawn new damage
        self.time_to_next_damage -= game_speed
        if self.time_to_next_damage <= 0:
            if self.damage_spawn_points:
                pos = random.choice(self.damage_spawn_points)
                # Ensure no damage spawns on top of another
                if not any(np.array_equal(site['pos'], pos) for site in self.damage_sites):
                    self.damage_sites.append({'pos': pos, 'health': 100, 'max_health': 100, 'pulse': random.random() * math.pi * 2})
                    # sfx: damage_spawn.wav
            
            # Difficulty scaling: damage appears faster over time
            self.damage_spawn_cooldown_base = max(1 * self.FPS, self.damage_spawn_cooldown_base * 0.995)
            self.time_to_next_damage = self.damage_spawn_cooldown_base

        # Apply damage to cell integrity
        integrity_lost = len(self.damage_sites) * 0.01 * game_speed
        self.cell_integrity = max(0, self.cell_integrity - integrity_lost)
        return integrity_lost

    def _update_nanobots(self, game_speed):
        repair_reward = 0
        completed_repairs = 0
        
        damage_positions = [tuple(site['pos']) for site in self.damage_sites]

        for bot in self.nanobots:
            if bot['state'] == 'idle':
                path = self._find_path_bfs(bot['grid_pos'], damage_positions)
                if path:
                    bot['state'] = 'moving'
                    bot['path'] = path
            
            if bot['state'] == 'moving':
                if not bot['path']:
                    bot['state'] = 'idle'
                    continue
                
                target_grid_pos = bot['path'][0]
                target_pixel_pos = self._get_pixel_pos(target_grid_pos)
                
                direction = target_pixel_pos - bot['pos']
                distance = np.linalg.norm(direction)
                
                if distance < 1:
                    bot['grid_pos'] = bot['path'].pop(0)
                    if not bot['path']:
                        bot['state'] = 'repairing'
                else:
                    move_speed = 3.0 * game_speed
                    bot['pos'] += (direction / distance) * min(move_speed, distance)

            if bot['state'] == 'repairing':
                # Find the damage site at the bot's location
                target_site = None
                bot_pixel_pos_tuple = tuple(self._get_pixel_pos(bot['grid_pos']))
                for site in self.damage_sites:
                    if tuple(site['pos']) == bot_pixel_pos_tuple:
                        target_site = site
                        break
                
                if target_site:
                    repair_amount = 0.5 * game_speed * self.nanobot_repair_speed_multiplier
                    target_site['health'] -= repair_amount
                    repair_reward += 0.1 # Continuous reward for repairing
                    # sfx: repairing_loop.wav (continuous)
                    if target_site['health'] <= 0:
                        self.damage_sites.remove(target_site)
                        bot['state'] = 'idle'
                        completed_repairs += 1
                        # sfx: repair_complete.wav
                else:
                    # Damage was repaired by another bot
                    bot['state'] = 'idle'
        
        return repair_reward, completed_repairs

    # --- Rendering and Observation ---

    def _get_observation(self):
        # Apply time warp visual effect
        y_offset = math.sin(self.steps * 0.2) * 3 if self.time_warp_active and self.chrono_energy > 0 else 0
        
        self.screen.fill(self.COLOR_BG)
        self._render_background(y_offset)
        self._render_scaffolding(y_offset)
        self._render_damage(y_offset)
        self._render_nanobots(y_offset)
        self._render_cursor(y_offset)
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, y_offset):
        # Draw cell wall
        wall_rect = pygame.Rect(20, 20, self.WIDTH - 40, self.HEIGHT - 40)
        pygame.draw.rect(self.screen, self.COLOR_CELL_WALL, wall_rect.move(0, y_offset), border_radius=30)
        pygame.draw.rect(self.screen, self.COLOR_CELL_WALL_OUTLINE, wall_rect.move(0, y_offset), width=3, border_radius=30)

    def _render_scaffolding(self, y_offset):
        for pos, connections in self.scaffolding_nodes.items():
            p1 = self._get_pixel_pos(pos)
            for neighbor_pos in connections:
                p2 = self._get_pixel_pos(neighbor_pos)
                pygame.draw.aaline(self.screen, self.COLOR_SCAFFOLD, p1 + [0, y_offset], p2 + [0, y_offset])
        
        for pos in self.scaffolding_nodes:
            pixel_pos = self._get_pixel_pos(pos)
            pygame.gfxdraw.filled_circle(self.screen, int(pixel_pos[0]), int(pixel_pos[1] + y_offset), 4, self.COLOR_SCAFFOLD_GLOW)
            pygame.gfxdraw.aacircle(self.screen, int(pixel_pos[0]), int(pixel_pos[1] + y_offset), 4, self.COLOR_SCAFFOLD_GLOW)

    def _render_damage(self, y_offset):
        for site in self.damage_sites:
            pos = site['pos']
            pulse_radius = 5 + 3 * math.sin(self.steps * 0.1 + site['pulse'])
            # Draw pulsating red damage
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1] + y_offset), int(pulse_radius), self.COLOR_DAMAGE)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1] + y_offset), int(pulse_radius), self.COLOR_DAMAGE)
            # Draw repair progress
            if site['health'] < site['max_health']:
                progress_angle = 360 * (1 - site['health'] / site['max_health'])
                if progress_angle > 0:
                    rect = pygame.Rect(pos[0] - pulse_radius, pos[1] - pulse_radius + y_offset, pulse_radius*2, pulse_radius*2)
                    pygame.draw.arc(self.screen, self.COLOR_REPAIR, rect, 0, math.radians(progress_angle), 3)

    def _render_nanobots(self, y_offset):
        for bot in self.nanobots:
            pos = bot['pos']
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1] + y_offset), 6, self.COLOR_NANO_GLOW)
            # Main body
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1] + y_offset), 4, self.COLOR_NANO)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1] + y_offset), 4, self.COLOR_NANO)

    def _render_cursor(self, y_offset):
        pos = self.cursor_pos
        size = 10
        points = [
            (pos[0], pos[1] - size + y_offset),
            (pos[0] + size, pos[1] + size + y_offset),
            (pos[0] - size, pos[1] + size + y_offset)
        ]
        pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_CURSOR)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - 120, 10))

        # Steps/Time
        time_left = (self.MAX_STEPS - self.steps) // self.FPS
        time_text = self.font_ui.render(f"Time: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - 120, 35))

        # Cell Integrity
        integrity_text = self.font_ui.render(f"Integrity: {int(self.cell_integrity)}%", True, self.COLOR_TEXT)
        self.screen.blit(integrity_text, (10, 10))
        pygame.draw.rect(self.screen, (50,50,50), (10, 35, 200, 15))
        if self.cell_integrity > 0:
            pygame.draw.rect(self.screen, self.COLOR_INTEGRITY_BAR, (10, 35, 2 * self.cell_integrity, 15))

        # Chrono Energy
        chrono_text = self.font_ui.render("Chrono-Energy", True, self.COLOR_TEXT)
        self.screen.blit(chrono_text, (10, self.HEIGHT - 30))
        pygame.draw.rect(self.screen, (50,50,50), (130, self.HEIGHT - 30, 200, 15))
        if self.chrono_energy > 0:
            pygame.draw.rect(self.screen, self.COLOR_CHRONO_BAR, (130, self.HEIGHT - 30, 2 * self.chrono_energy, 15))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text = "FAILURE" if self.cell_integrity <= 0 else "TIME UP"
        game_over_text = self.font_game_over.render(text, True, self.COLOR_DAMAGE)
        text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(game_over_text, text_rect)

    # --- Helper Functions ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cell_integrity": self.cell_integrity,
            "repairs_completed": self.repairs_completed
        }

    def _check_termination(self):
        return self.cell_integrity <= 0

    def _define_damage_spawn_points(self):
        self.damage_spawn_points = []
        wall_rect = pygame.Rect(20, 20, self.WIDTH - 40, self.HEIGHT - 40)
        for x in range(wall_rect.left, wall_rect.right, 20):
            self.damage_spawn_points.append(np.array([x, wall_rect.top]))
            self.damage_spawn_points.append(np.array([x, wall_rect.bottom]))
        for y in range(wall_rect.top, wall_rect.bottom, 20):
            self.damage_spawn_points.append(np.array([wall_rect.left, y]))
            self.damage_spawn_points.append(np.array([wall_rect.right, y]))

    def _get_grid_pos(self, pixel_pos):
        return (
            int(pixel_pos[0] / self.scaffold_grid_size),
            int(pixel_pos[1] / self.scaffold_grid_size)
        )

    def _get_pixel_pos(self, grid_pos):
        return np.array([
            grid_pos[0] * self.scaffold_grid_size + self.scaffold_grid_size / 2,
            grid_pos[1] * self.scaffold_grid_size + self.scaffold_grid_size / 2
        ])
    
    def _create_nanobot(self):
        hub_grid_pos = self._get_grid_pos(np.array([self.WIDTH / 2, self.HEIGHT / 2]))
        return {
            'pos': self._get_pixel_pos(hub_grid_pos) + self.np_random.random(2) * 5 - 2.5,
            'grid_pos': hub_grid_pos,
            'state': 'idle', # idle, moving, repairing
            'path': []
        }

    def _find_path_bfs(self, start_node_pos, target_pixel_positions):
        if not target_pixel_positions or not self.scaffolding_nodes:
            return None

        target_node_positions = {self._get_grid_pos(p) for p in target_pixel_positions}
        
        queue = deque([[start_node_pos]])
        visited = {start_node_pos}

        while queue:
            path = queue.popleft()
            node = path[-1]

            if node in target_node_positions:
                return path[1:] # Return path without the starting node

            for neighbor in self.scaffolding_nodes.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
        return None

# Example usage to run and visualize the environment
if __name__ == '__main__':
    # This block is for human play and visualization.
    # It will not be executed by the test suite.
    
    # Un-comment the line below to run with a display window
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display window
    pygame.display.set_caption("Nanobot Cell Repair")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    done = False
    total_reward = 0
    
    # Use keyboard for human play
    key_map = {
        pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4,
    }
    
    while not done:
        # Human input
        movement_action = 0 # No-op
        space_action = 0
        shift_action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds before closing
            done = True
            
    pygame.quit()