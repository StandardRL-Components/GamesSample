import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:45:22.641827
# Source Brief: brief_02987.md
# Brief Index: 2987
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a colony of cells.
    The goal is to grow the colony to 50 cells by consuming resources.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]`: Movement (0: none, 1: up, 2: down, 3: left, 4: right)
    - `actions[1]`: Space button (unused)
    - `actions[2]`: Shift button (unused)

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game state.

    **Reward Structure:**
    - +0.1 for each resource consumed.
    - +1.0 for each new cell division.
    - +100 for winning (reaching 50 cells).
    - -100 for losing (all cells die).
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control a colony of cells, moving them to consume resources and divide. "
        "Grow your population to 50 to win, but be careful, as moving into an empty space will starve your cells."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the entire cell colony. "
        "Move onto a resource to consume it and grow."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 20
    GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
    GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE
    
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 45, 50)
    COLOR_CELL = (0, 255, 127) # Spring Green
    COLOR_CELL_GLOW = (127, 255, 212) # Aquamarine
    COLOR_RESOURCE = (0, 191, 255) # Deep Sky Blue
    COLOR_TEXT = (230, 230, 230)

    WIN_POPULATION = 50
    MAX_STEPS = 1000
    INITIAL_RESOURCE_DENSITY = 0.7
    CELL_DIVISION_ENERGY = 30
    
    # Assumes 30 FPS for visual interpolation and regeneration timing
    FRAME_RATE = 30 
    RESOURCE_REGEN_TIME_SECONDS = 10 
    RESOURCE_REGEN_FRAMES = RESOURCE_REGEN_TIME_SECONDS * FRAME_RATE
    VISUAL_LERP_RATE = 0.25 # How fast visuals catch up to logic state

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode

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
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # Initialize state variables
        self.cells = []
        self.particles = []
        self.resources = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.int8)
        self.resource_regen_timers = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=np.int32)
        self.steps = 0
        self.score = 0
        self.game_over = False
        
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cells = []
        self.particles = []

        # Initialize one cell in the center
        center_x, center_y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self._create_cell(pygame.Vector2(center_x, center_y))
        
        # Initialize resources
        self.resources = (self.np_random.random((self.GRID_WIDTH, self.GRID_HEIGHT)) < self.INITIAL_RESOURCE_DENSITY).astype(np.int8)
        self.resource_regen_timers.fill(0)
        # Ensure the starting cell has a resource
        self.resources[center_x, center_y] = 1

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        self.steps += 1

        # --- Game Logic ---
        
        # 1. Handle Movement and Resource Consumption
        if movement != 0:
            move_vector = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            move_vector = pygame.Vector2(move_vector)
            
            newly_dead_cells = []
            for cell in self.cells:
                target_pos = cell['grid_pos'] + move_vector
                tx, ty = int(target_pos.x), int(target_pos.y)

                # Check boundaries
                if not (0 <= tx < self.GRID_WIDTH and 0 <= ty < self.GRID_HEIGHT):
                    continue # Cell hits a wall, stays put

                # Check for resource at target
                if self.resources[tx, ty] == 1:
                    # Consume resource
                    self.resources[tx, ty] = 0
                    self.resource_regen_timers[tx, ty] = self.RESOURCE_REGEN_FRAMES
                    cell['grid_pos'] = target_pos
                    cell['energy'] += 1
                    reward += 0.1
                    # Sound: resource consume blip
                else:
                    # Starve
                    newly_dead_cells.append(cell)
                    # Sound: cell death sizzle
            
            # Remove starved cells
            if newly_dead_cells:
                self.cells = [c for c in self.cells if c not in newly_dead_cells]

        # 2. Handle Cell Division
        newly_born_cells = []
        for cell in self.cells:
            if cell['energy'] >= self.CELL_DIVISION_ENERGY:
                cell['energy'] = 0
                new_cell = self._create_cell(cell['grid_pos'].copy())
                newly_born_cells.append(new_cell)
                reward += 1.0
                self._create_particles(cell['visual_pos'], self.COLOR_CELL, 15)
                # Sound: cell division pop
        self.cells.extend(newly_born_cells)

        # 3. Handle Resource Regeneration
        regen_mask = self.resource_regen_timers > 0
        self.resource_regen_timers[regen_mask] -= 1
        self.resources[self.resource_regen_timers == 0] = 1

        # --- Termination Check ---
        terminated = False
        truncated = False
        if len(self.cells) >= self.WIN_POPULATION:
            reward += 100
            terminated = True
            self.game_over = True
        elif not self.cells:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time/step limits
            self.game_over = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _create_cell(self, grid_pos):
        cell = {
            'grid_pos': grid_pos,
            'visual_pos': grid_pos * self.GRID_SIZE + pygame.Vector2(self.GRID_SIZE / 2),
            'energy': 0,
            'radius': self.GRID_SIZE * 0.4
        }
        if not hasattr(self, 'cells') or not self.cells: # First cell
             self.cells = [cell]
        return cell

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': random.randint(15, 30), # frames
                'color': color
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw resources
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                if self.resources[x, y] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_RESOURCE, rect.inflate(-self.GRID_SIZE*0.5, -self.GRID_SIZE*0.5))
                elif self.resource_regen_timers[x,y] > 0:
                    # Fading in effect
                    progress = 1.0 - (self.resource_regen_timers[x,y] / self.RESOURCE_REGEN_FRAMES)
                    alpha = int(progress * 255)
                    color = (*self.COLOR_RESOURCE, alpha)
                    
                    s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
                    s.fill((0,0,0,0))
                    pygame.draw.rect(s, color, s.get_rect().inflate(-self.GRID_SIZE*0.5, -self.GRID_SIZE*0.5))
                    self.screen.blit(s, rect.topleft)

        # Update and draw particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 30.0))
                pygame.draw.circle(self.screen, (*p['color'], alpha), p['pos'], p['life'] * 0.1)

        # Update and draw cells
        for cell in self.cells:
            # Interpolate visual position towards logical grid position
            target_visual_pos = cell['grid_pos'] * self.GRID_SIZE + pygame.Vector2(self.GRID_SIZE / 2)
            cell['visual_pos'] = cell['visual_pos'].lerp(target_visual_pos, self.VISUAL_LERP_RATE)
            
            pos = (int(cell['visual_pos'].x), int(cell['visual_pos'].y))
            radius = int(cell['radius'])

            # Draw glow effect
            for i in range(radius // 2, 0, -2):
                alpha = 50 * (1 - (i / (radius // 2)))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + i, (*self.COLOR_CELL_GLOW, int(alpha)))
            
            # Draw main cell body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_CELL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_CELL)


    def _render_ui(self):
        pop_text = f"POPULATION: {len(self.cells):02d} / {self.WIN_POPULATION}"
        step_text = f"STEPS: {self.steps:04d} / {self.MAX_STEPS}"
        score_text = f"SCORE: {self.score:.1f}"

        pop_surf = self.font_main.render(pop_text, True, self.COLOR_TEXT)
        step_surf = self.font_small.render(step_text, True, self.COLOR_TEXT)
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        
        self.screen.blit(pop_surf, (10, 10))
        self.screen.blit(step_surf, (10, 35))
        self.screen.blit(score_surf, (10, 50))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "VICTORY!" if len(self.cells) >= self.WIN_POPULATION else "EXTINCTION"
            status_surf = pygame.font.SysFont("Consolas", 60, bold=True).render(status_text, True, self.COLOR_CELL if len(self.cells) >= self.WIN_POPULATION else (255, 50, 50))
            status_rect = status_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(status_surf, status_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "population": len(self.cells)
        }
    
    def close(self):
        pygame.quit()
        
# Example usage:
if __name__ == "__main__":
    # Un-set the dummy driver if we are running this directly
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Use arrow keys to move. Q to quit. R to reset.
    
    running = True
    done = False
    total_reward = 0
    
    # Pygame window for human play
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cell Colony")
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # No-op
    
    while running:
        # Reset action to no-op, it will be set by key presses
        action = np.array([0, 0, 0])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                    total_reward = 0
                
                # Manual control mapping
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4

        # Only step if an action was taken (or if the game is over and we just need to render)
        if not done and np.any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        # Render the observation from the environment
        # The main loop needs to get an up-to-date observation even if no action is taken
        # This is especially true for the final frame after the game is done.
        current_obs = env._get_observation()
        surf = pygame.surfarray.make_surface(np.transpose(current_obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Limit frame rate for human play
        
    env.close()
    pygame.quit()