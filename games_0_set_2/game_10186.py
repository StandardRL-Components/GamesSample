import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:03:41.247761
# Source Brief: brief_00186.md
# Brief Index: 186
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An expert-crafted Gymnasium environment for a visually polished underwater ROV puzzle game.

    The agent controls an ROV to collect resources while avoiding mines.
    The core mechanic involves drawing "path cards" that determine the ROV's
    possible moves, adding a strategic planning layer to the gameplay.

    Visuals and game feel are prioritized to create an engaging experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control an underwater ROV to collect resources and avoid mines. "
        "Strategically use path cards to navigate the grid and clear the level."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move along the path shown on your current card. "
        "Press space to draw a new path card."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_W, SCREEN_H = 640, 400
    GRID_W, GRID_H = 16, 10
    CELL_SIZE = 40
    MAX_STEPS = 2000
    NUM_RESOURCES = 10
    MINE_DENSITY = 0.10
    ROV_LERP_RATE = 0.5  # Controls visual movement smoothness

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (20, 40, 80)
    COLOR_ROV = (255, 255, 255)
    COLOR_ROV_GLOW = (200, 200, 255)
    COLOR_RESOURCE = (255, 220, 0)
    COLOR_MINE = (255, 50, 50)
    COLOR_PATH_PREVIEW = (100, 150, 255, 150)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    # --- Path Card Definitions ---
    # Cards define which directions are available for movement.
    # 0: Straight (Vertical), 1: Straight (Horizontal), 2: L-bend, 3: T-junction, 4: Cross
    CARD_SHAPES = {
        0: {"name": "I-Vert", "moves": [(0, -1), (0, 1)]},
        1: {"name": "I-Horiz", "moves": [(-1, 0), (1, 0)]},
        2: {"name": "L-Bend", "moves": [(0, -1), (1, 0)]},
        3: {"name": "T-Junc", "moves": [(0, -1), (1, 0), (-1, 0)]},
        4: {"name": "Cross", "moves": [(0, -1), (0, 1), (-1, 0), (1, 0)]},
    }
    # Add rotations for more variety
    CARD_SHAPES[5] = {"name": "L-Bend", "moves": [(0, -1), (-1, 0)]}
    CARD_SHAPES[6] = {"name": "L-Bend", "moves": [(0, 1), (1, 0)]}
    CARD_SHAPES[7] = {"name": "L-Bend", "moves": [(0, 1), (-1, 0)]}
    CARD_SHAPES[8] = {"name": "T-Junc", "moves": [(0, 1), (1, 0), (-1, 0)]}
    CARD_SHAPES[9] = {"name": "T-Junc", "moves": [(0, -1), (0, 1), (1, 0)]}
    CARD_SHAPES[10] = {"name": "T-Junc", "moves": [(0, -1), (0, 1), (-1, 0)]}

    # --- Action to Direction Mapping ---
    ACTION_TO_DIR = {
        1: (0, -1),  # Up
        2: (0, 1),   # Down
        3: (-1, 0),  # Left
        4: (1, 0),   # Right
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont('Consolas', 24, bold=True)
        self.font_card = pygame.font.SysFont('Consolas', 18)

        # --- Game State Variables ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.rov_pos = [0, 0]
        self.rov_visual_pos = [0.0, 0.0]
        self.resources = []
        self.mines = []
        self.particles = []
        self.current_card_id = 0
        self.total_resources_initial = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        # --- Procedural Level Generation ---
        all_coords = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        random.shuffle(all_coords)

        # Place ROV
        self.rov_pos = [1, self.GRID_H // 2]
        self.rov_visual_pos = [
            self.rov_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2,
            self.rov_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        ]
        if tuple(self.rov_pos) in all_coords:
            all_coords.remove(tuple(self.rov_pos))

        # Place Resources
        self.resources = []
        for _ in range(self.NUM_RESOURCES):
            if not all_coords: break
            self.resources.append(list(all_coords.pop()))
        self.total_resources_initial = len(self.resources)

        # Place Mines
        self.mines = []
        num_mines = int(self.GRID_W * self.GRID_H * self.MINE_DENSITY)
        for _ in range(num_mines):
            if not all_coords: break
            self.mines.append(list(all_coords.pop()))

        # Draw first card
        self._draw_card()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for taking a step

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Action Handling ---
        # Action priority: Draw Card > Move ROV
        if space_pressed:
            # Drawing a card is a full action for a turn
            self._draw_card()
            reward += -0.05 # small cost to draw a card
            # sfx: card_draw.wav
        elif movement in self.ACTION_TO_DIR:
            reward = self._execute_move(movement, reward)
        
        # --- Update Game State ---
        self._update_visuals()
        self._update_particles()
        
        # --- Collision Detection ---
        # Resource collection
        collected_resource = None
        for res in self.resources:
            if res == self.rov_pos:
                collected_resource = res
                break
        if collected_resource:
            self.resources.remove(collected_resource)
            self.score += 1
            reward += 1.0
            self._create_sparkles(self._grid_to_pixel(collected_resource))
            # sfx: resource_collect.wav

        # Mine collision
        terminated = False
        if self.rov_pos in self.mines:
            reward -= 10.0
            terminated = True
            self.game_over = True
            self._create_explosion(self._grid_to_pixel(self.rov_pos))
            # sfx: explosion.wav
        
        # --- Check Other Termination Conditions ---
        truncated = False
        if not terminated:
            if self.steps >= self.MAX_STEPS:
                truncated = True
                self.game_over = True
            elif not self.resources: # All resources collected
                reward += 50.0
                terminated = True
                self.game_over = True
                # sfx: victory.wav
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _execute_move(self, movement_action, reward):
        target_dir = self.ACTION_TO_DIR[movement_action]
        card_moves = self.CARD_SHAPES[self.current_card_id]["moves"]

        if target_dir in card_moves:
            new_pos = [self.rov_pos[0] + target_dir[0], self.rov_pos[1] + target_dir[1]]
            
            # Boundary check
            if 0 <= new_pos[0] < self.GRID_W and 0 <= new_pos[1] < self.GRID_H:
                self.rov_pos = new_pos
                # sfx: rov_move.wav
            else:
                reward -= 0.1 # Penalty for invalid move attempt (bumping wall)
        else:
            reward -= 0.1 # Penalty for invalid move attempt (wrong card path)
        return reward

    def _draw_card(self):
        self.current_card_id = self.np_random.integers(0, len(self.CARD_SHAPES))

    def _update_visuals(self):
        # Interpolate ROV visual position for smooth movement
        target_pixel_pos = self._grid_to_pixel(self.rov_pos)
        self.rov_visual_pos[0] += (target_pixel_pos[0] - self.rov_visual_pos[0]) * self.ROV_LERP_RATE
        self.rov_visual_pos[1] += (target_pixel_pos[1] - self.rov_visual_pos[1]) * self.ROV_LERP_RATE

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
            "resources_remaining": len(self.resources),
            "rov_pos": self.rov_pos,
        }

    # =================================================================
    # --- Rendering Methods ---
    # =================================================================

    def _render_game(self):
        self._render_grid()
        self._render_mines()
        self._render_resources()
        self._render_path_preview()
        self._render_particles()
        self._render_rov()

    def _render_grid(self):
        for x in range(self.GRID_W + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_H))
        for y in range(self.GRID_H + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_W, py))

    def _render_mines(self):
        for mine_pos in self.mines:
            pixel_pos = self._grid_to_pixel(mine_pos)
            radius = self.CELL_SIZE * 0.35
            points = []
            for i in range(8):
                angle = math.pi / 8 + i * math.pi / 4
                points.append((
                    int(pixel_pos[0] + radius * math.cos(angle)),
                    int(pixel_pos[1] + radius * math.sin(angle))
                ))
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_MINE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_MINE)

    def _render_resources(self):
        for res_pos in self.resources:
            pixel_pos = self._grid_to_pixel(res_pos)
            radius = int(self.CELL_SIZE * 0.3)
            pygame.gfxdraw.aacircle(self.screen, int(pixel_pos[0]), int(pixel_pos[1]), radius, self.COLOR_RESOURCE)
            pygame.gfxdraw.filled_circle(self.screen, int(pixel_pos[0]), int(pixel_pos[1]), radius, self.COLOR_RESOURCE)

    def _render_path_preview(self):
        if self.game_over: return
        start_pos = self._grid_to_pixel(self.rov_pos)
        card = self.CARD_SHAPES[self.current_card_id]
        for move in card["moves"]:
            end_pos_grid = (self.rov_pos[0] + move[0], self.rov_pos[1] + move[1])
            if 0 <= end_pos_grid[0] < self.GRID_W and 0 <= end_pos_grid[1] < self.GRID_H:
                end_pos = self._grid_to_pixel(end_pos_grid)
                pygame.draw.aaline(self.screen, self.COLOR_PATH_PREVIEW, start_pos, end_pos, True)

    def _render_rov(self):
        pos = (int(self.rov_visual_pos[0]), int(self.rov_visual_pos[1]))
        size = int(self.CELL_SIZE * 0.6)
        
        # Glow effect
        for i in range(10, 0, -2):
            alpha = 50 - i * 5
            glow_color = (*self.COLOR_ROV_GLOW, alpha)
            radius = size // 2 + i
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, glow_color)
        
        # Main body
        rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_ROV, rect, border_radius=3)

    def _render_ui(self):
        # --- Score and Steps ---
        score_text = f"SCORE: {self.score}"
        steps_text = f"STEPS: {self.steps}/{self.MAX_STEPS}"
        self._draw_text(score_text, (15, 10), self.font_main)
        self._draw_text(steps_text, (15, 35), self.font_main)
        
        # --- Card Display ---
        card_area_rect = pygame.Rect(self.SCREEN_W - 140, 10, 130, 80)
        pygame.draw.rect(self.screen, self.COLOR_GRID, card_area_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PATH_PREVIEW, card_area_rect, width=1, border_radius=5)
        
        self._draw_text("CURRENT CARD", (card_area_rect.centerx, card_area_rect.top + 10), self.font_card, align="center")
        
        # Draw miniature card
        card = self.CARD_SHAPES[self.current_card_id]
        center_x, center_y = card_area_rect.centerx, card_area_rect.centery + 10
        pygame.draw.circle(self.screen, self.COLOR_ROV, (center_x, center_y), 4)
        for move in card["moves"]:
            end_x = center_x + move[0] * 15
            end_y = center_y + move[1] * 15
            pygame.draw.aaline(self.screen, self.COLOR_PATH_PREVIEW, (center_x, center_y), (end_x, end_y))

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, align="left"):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        
        text_rect = text_surf.get_rect()
        if align == "center":
            text_rect.center = pos
        else: # left
            text_rect.topleft = pos
        
        shadow_rect = text_rect.copy()
        shadow_rect.topleft = (text_rect.left + 1, text_rect.top + 1)
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    # =================================================================
    # --- Particle System Methods ---
    # =================================================================

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.1)
    
    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            if p['shape'] == 'circle':
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['size']), color)
            elif p['shape'] == 'rect':
                rect = pygame.Rect(p['pos'][0] - p['size']/2, p['pos'][1] - p['size']/2, p['size'], p['size'])
                pygame.draw.rect(self.screen, p['color'], rect) # Simple rect for flash
    
    def _create_sparkles(self, pos):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'size': random.uniform(2, 5),
                'color': self.COLOR_RESOURCE,
                'shape': 'circle'
            })

    def _create_explosion(self, pos):
        # Big flash
        self.particles.append({
            'pos': list(pos), 'vel': [0, 0], 'life': 8, 'max_life': 8,
            'size': self.CELL_SIZE * 3, 'color': (255, 100, 100, 100), 'shape': 'circle'
        })
        # Debris
        for _ in range(40):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            life = random.randint(20, 40)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'size': random.uniform(3, 6),
                'color': self.COLOR_MINE,
                'shape': 'circle'
            })

    # =================================================================
    # --- Helper & Validation Methods ---
    # =================================================================

    def _grid_to_pixel(self, grid_pos):
        px = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        py = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        return (px, py)
        
    def close(self):
        pygame.quit()

# Example of how to use the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Loop ---
    # Controls:
    # Arrows: Move ROV (if allowed by card)
    # Space: Draw new card
    # R: Reset environment
    # Q: Quit
    
    running = True
    
    # The main script needs to set the video driver to something other than dummy
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.display.set_caption("ROV Puzzle Environment")
    screen_display = pygame.display.set_mode((GameEnv.SCREEN_W, GameEnv.SCREEN_H))
    
    action = [0, 0, 0] # [movement, space, shift]

    while running:
        # The step logic is designed for one action at a time.
        # For human play, we collect an action and then step.
        current_action = [0, 0, 0] # Reset action for this frame
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    continue # Skip stepping on reset frame
                
                # Map key presses to an action for the *next* step
                if event.key == pygame.K_UP: current_action[0] = 1
                elif event.key == pygame.K_DOWN: current_action[0] = 2
                elif event.key == pygame.K_LEFT: current_action[0] = 3
                elif event.key == pygame.K_RIGHT: current_action[0] = 4
                elif event.key == pygame.K_SPACE: current_action[1] = 1

                # Step the environment with the collected action
                obs, reward, terminated, truncated, info = env.step(current_action)

                if terminated or truncated:
                    print(f"Episode finished. Score: {info['score']}, Steps: {info['steps']}")
                    # To auto-reset after an episode ends:
                    # obs, info = env.reset() 

        # If no key was pressed, we can optionally send a no-op
        # This part is commented out as the current human-play loop only
        # steps when a key is pressed.
        # if not any(current_action):
        #     obs, reward, terminated, truncated, info = env.step([0, 0, 0])

        # Render the current observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit FPS for human play

    pygame.quit()
    env.close()