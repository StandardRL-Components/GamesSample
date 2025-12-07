import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:53:15.863467
# Source Brief: brief_01969.md
# Brief Index: 1969
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple, deque
import copy

# Define a structure type for better organization
Structure = namedtuple("Structure", ["name", "cost", "color", "unlock_pop"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Build a microscopic city on a petri dish. Manage resources like nutrients and water to grow your population while handling waste."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press 'space' to build and 'shift' to cycle structures. Hold 'shift' + 'space' to rewind."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 32, 20
    CELL_SIZE = WIDTH // GRID_COLS
    MAX_STEPS = 10000
    WIN_POPULATION = 1000
    REWIND_HISTORY_LENGTH = 50

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_NUTRIENT_SRC = (20, 80, 40)
    COLOR_WATER_SRC = (20, 40, 100)
    COLOR_WASTE = (255, 60, 60)
    
    STRUCTURE_TYPES = {
        0: Structure(name="Harvester", cost={'nutrients': 10, 'water': 0}, color=(0, 255, 100), unlock_pop=0),
        1: Structure(name="Water Pump", cost={'nutrients': 10, 'water': 0}, color=(50, 150, 255), unlock_pop=0),
        2: Structure(name="Habitation", cost={'nutrients': 20, 'water': 10}, color=(100, 200, 255), unlock_pop=0),
        3: Structure(name="Recycler", cost={'nutrients': 50, 'water': 20}, color=(255, 150, 0), unlock_pop=500),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.paused = False
        self.grid = None
        self.sources = None
        self.resources = None
        self.waste_level = 0
        self.population = 0
        self.cursor_pos = None
        self.visual_cursor_pos = None
        self.selected_structure_idx = 0
        self.unlocked_structures = []
        self.particles = []
        self.last_action = np.array([0, 0, 0])
        self.state_history = deque(maxlen=self.REWIND_HISTORY_LENGTH)
        self.win_message = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.paused = False
        self.win_message = ""
        
        self.grid = np.full((self.GRID_COLS, self.GRID_ROWS), -1, dtype=int)
        self.sources = np.full((self.GRID_COLS, self.GRID_ROWS), -1, dtype=int) # 0=nutrient, 1=water
        
        self.resources = {'nutrients': 100, 'water': 50}
        self.waste_level = 0
        self.population = 0
        
        self.cursor_pos = np.array([self.GRID_COLS // 2, self.GRID_ROWS // 2])
        self.visual_cursor_pos = self.cursor_pos.astype(float) * self.CELL_SIZE
        
        self.selected_structure_idx = 0
        self._update_unlocked_structures()
        
        self.particles = []
        self.last_action = np.array([0, 0, 0])
        
        self.state_history.clear()

        # Generate sources
        num_nutrient_sources = self.np_random.integers(4, 7)
        num_water_sources = self.np_random.integers(3, 6)
        for _ in range(num_nutrient_sources):
            x, y = self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS)
            self.sources[x, y] = 0
        for _ in range(num_water_sources):
            x, y = self.np_random.integers(0, self.GRID_COLS), self.np_random.integers(0, self.GRID_ROWS)
            if self.sources[x, y] == -1: # Avoid overlap
                self.sources[x, y] = 1

        self._save_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_action[1]
        shift_press = shift_held and not self.last_action[2]

        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Actions ---
        if space_held and shift_held: # Rewind action
            reward += self._rewind()
        else:
            self._save_state() # Save state before any changes
            
            if shift_press: # Cycle structure
                self.selected_structure_idx = (self.selected_structure_idx + 1) % len(self.unlocked_structures)
            
            if space_press: # Place structure
                reward += self._place_structure()

            # Move cursor
            if movement == 1: self.cursor_pos[1] -= 1
            elif movement == 2: self.cursor_pos[1] += 1
            elif movement == 3: self.cursor_pos[0] -= 1
            elif movement == 4: self.cursor_pos[0] += 1
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

            # --- Update Game Simulation ---
            self.steps += 1
            sim_reward = self._update_simulation()
            reward += sim_reward

        # Update last action state for press detection
        self.last_action = action
        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
            self.win_message = "TIME LIMIT REACHED"
            self.score -= 50

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _save_state(self):
        state = {
            'steps': self.steps,
            'score': self.score,
            'grid': copy.deepcopy(self.grid),
            'resources': copy.deepcopy(self.resources),
            'waste_level': self.waste_level,
            'population': self.population,
            'unlocked_structures': copy.deepcopy(self.unlocked_structures),
            'selected_structure_idx': self.selected_structure_idx,
        }
        self.state_history.append(state)

    def _rewind(self):
        if len(self.state_history) > 1:
            self.state_history.pop() # Remove current state
            last_state = self.state_history[-1] # Get previous state
            
            self.steps = last_state['steps']
            self.score = last_state['score']
            self.grid = copy.deepcopy(last_state['grid'])
            self.resources = copy.deepcopy(last_state['resources'])
            self.waste_level = last_state['waste_level']
            self.population = last_state['population']
            self.unlocked_structures = copy.deepcopy(last_state['unlocked_structures'])
            self.selected_structure_idx = last_state['selected_structure_idx']
            # SFX: Rewind sound
            return -0.5 # Small penalty for rewinding
        return 0

    def _update_simulation(self):
        pop_before = self.population
        reward = 0
        
        # Resource generation and consumption
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                struct_id = self.grid[x, y]
                if struct_id == -1: continue

                # Harvester on nutrient source
                if struct_id == 0 and self.sources[x, y] == 0:
                    self.resources['nutrients'] += 1
                    self._create_particle((x, y), self.STRUCTURE_TYPES[0].color, 'resource')
                # Water Pump on water source
                elif struct_id == 1 and self.sources[x, y] == 1:
                    self.resources['water'] += 1
                    self._create_particle((x, y), self.STRUCTURE_TYPES[1].color, 'resource')
                # Habitation
                elif struct_id == 2:
                    if self.resources['nutrients'] >= 1 and self.resources['water'] >= 0.5:
                        self.resources['nutrients'] -= 1
                        self.resources['water'] -= 0.5
                        self.population += 1
                        self.waste_level += 1
                        self._create_particle((x, y), self.COLOR_WASTE, 'waste')
                # Recycler
                elif struct_id == 3:
                    if self.waste_level >= 2:
                        self.waste_level -= 2
                        self.resources['nutrients'] += 0.5

        # Cap resources
        self.resources['nutrients'] = min(self.resources['nutrients'], 9999)
        self.resources['water'] = min(self.resources['water'], 9999)
        
        # Rewards from simulation
        pop_gain = self.population - pop_before
        if pop_gain > 0:
            reward += pop_gain * 0.1
        reward -= self.waste_level * 0.001 # Small continuous penalty for waste

        # Check for unlocks
        unlocked_before = len(self.unlocked_structures)
        self._update_unlocked_structures()
        if len(self.unlocked_structures) > unlocked_before:
            reward += 10 # Big reward for unlocking something new
            # SFX: Unlock sound

        return reward

    def _update_unlocked_structures(self):
        self.unlocked_structures = [
            s_id for s_id, s_info in self.STRUCTURE_TYPES.items() 
            if self.population >= s_info.unlock_pop
        ]
        # Ensure selected index is valid
        if self.unlocked_structures and self.selected_structure_idx >= len(self.unlocked_structures):
            self.selected_structure_idx = 0

    def _place_structure(self):
        x, y = self.cursor_pos
        if self.grid[x, y] == -1 and self.unlocked_structures: # Can only build on empty cells
            struct_id = self.unlocked_structures[self.selected_structure_idx]
            structure = self.STRUCTURE_TYPES[struct_id]
            
            can_afford = all(self.resources[res] >= cost for res, cost in structure.cost.items())
            if can_afford:
                for res, cost in structure.cost.items():
                    self.resources[res] -= cost
                self.grid[x, y] = struct_id
                # SFX: Place structure
                return 5 # Reward for successful placement
        return -0.1 # Penalty for failed placement attempt

    def _check_termination(self):
        if self.game_over:
            return True

        if self.population >= self.WIN_POPULATION:
            self.game_over = True
            self.win_message = "VICTORY"
            self.score += 100
            return True
            
        # Loss condition: waste fills all available empty cells
        empty_cells = np.sum(self.grid == -1)
        if empty_cells > 0 and self.waste_level >= empty_cells * 10: # Arbitrary high waste factor
            self.game_over = True
            self.win_message = "WASTE OVERFLOW"
            self.score -= 100
            return True

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "population": self.population,
            "waste": self.waste_level,
            "nutrients": self.resources['nutrients'],
            "water": self.resources['water'],
        }

    def _render_game(self):
        # Draw grid and sources
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                
                # Draw sources as background circles
                source_type = self.sources[x, y]
                if source_type != -1:
                    color = self.COLOR_NUTRIENT_SRC if source_type == 0 else self.COLOR_WATER_SRC
                    pygame.draw.circle(self.screen, color, rect.center, self.CELL_SIZE // 2 - 2)

                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                # Draw structures
                struct_id = self.grid[x, y]
                if struct_id != -1:
                    struct_info = self.STRUCTURE_TYPES[struct_id]
                    pygame.draw.rect(self.screen, struct_info.color, rect.inflate(-4, -4))

        self._update_and_render_particles()
        self._render_cursor()

    def _render_cursor(self):
        target_pos = self.cursor_pos.astype(float) * self.CELL_SIZE
        # Interpolate for smooth movement
        self.visual_cursor_pos += (target_pos - self.visual_cursor_pos) * 0.4
        
        cursor_rect = pygame.Rect(self.visual_cursor_pos[0], self.visual_cursor_pos[1], self.CELL_SIZE, self.CELL_SIZE)
        
        # Draw glowing effect
        for i in range(4, 0, -1):
            glow_color = (*self.COLOR_CURSOR, 80 // i)
            pygame.draw.rect(self.screen, glow_color, cursor_rect.inflate(i*2, i*2), border_radius=3)
        
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=3)

    def _render_ui(self):
        # Top bar resources
        pop_text = self.font_small.render(f"POP: {int(self.population)}", True, self.COLOR_TEXT)
        waste_text = self.font_small.render(f"WASTE: {int(self.waste_level)}", True, self.COLOR_WASTE)
        nut_text = self.font_small.render(f"NUTR: {int(self.resources['nutrients'])}", True, self.STRUCTURE_TYPES[0].color)
        wat_text = self.font_small.render(f"WATER: {int(self.resources['water'])}", True, self.STRUCTURE_TYPES[1].color)
        
        self.screen.blit(pop_text, (10, 5))
        self.screen.blit(waste_text, (130, 5))
        self.screen.blit(nut_text, (270, 5))
        self.screen.blit(wat_text, (390, 5))

        # Selected structure
        if self.unlocked_structures:
            sel_id = self.unlocked_structures[self.selected_structure_idx]
            sel_struct = self.STRUCTURE_TYPES[sel_id]
            sel_text = self.font_small.render(f"BUILD: {sel_struct.name}", True, self.COLOR_TEXT)
            cost_text_str = f"Cost: {sel_struct.cost['nutrients']} N, {sel_struct.cost['water']} W"
            cost_text = self.font_small.render(cost_text_str, True, self.COLOR_TEXT)
            
            self.screen.blit(sel_text, (10, self.HEIGHT - 45))
            self.screen.blit(cost_text, (10, self.HEIGHT - 25))

        # Game over message
        if self.game_over and self.win_message:
            msg_surf = self.font_large.render(self.win_message, True, self.COLOR_CURSOR)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            pygame.draw.rect(self.screen, (0,0,0,150), msg_rect.inflate(20,20))
            self.screen.blit(msg_surf, msg_rect)

    def _create_particle(self, start_grid_pos, color, p_type):
        if len(self.particles) > 200: return # Limit particles
        
        start_pos = [p * self.CELL_SIZE + self.CELL_SIZE/2 for p in start_grid_pos]
        
        if p_type == 'resource':
            # Move towards UI bar
            end_pos = [random.uniform(250, 450), 10]
            velocity = [(end_pos[0] - start_pos[0]) / 60, (end_pos[1] - start_pos[1]) / 60]
        else: # waste
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]

        self.particles.append({
            'pos': start_pos,
            'vel': velocity,
            'color': color,
            'life': random.uniform(50, 80)
        })

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            
            alpha = max(0, min(255, int(p['life'] * 3)))
            color = (*p['color'], alpha)
            
            # Use gfxdraw for anti-aliased, alpha-blended circles
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(max(1, p['life'] / 20))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run when the environment is used by the training pipeline
    env = GameEnv()
    obs, info = env.reset()
    
    # Re-initialize pygame for display
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit()
    pygame.init()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Microscopic City Builder")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = np.array([movement, space_held, shift_held])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        # Display score for manual play
        score_text = env.font_small.render(f"SCORE: {total_reward:.2f}", True, (255, 255, 0))
        screen.blit(score_text, (520, 5))

        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {total_reward:.2f}, Info: {info}")
            pygame.time.wait(2000) # Pause before resetting
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()