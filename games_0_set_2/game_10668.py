import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:45:35.336906
# Source Brief: brief_00668.md
# Brief Index: 668
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a shape-shifting maze game.

    The agent controls a blob that can change between a square, circle, and
    triangle. It must navigate a maze with gates that only allow passage to
    the matching shape. The gates' required shape cycles every 10 seconds.
    The goal is to reach the green exit tile before the timer runs out.

    Visuals are minimalist and neon-themed for clarity and style.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a maze by changing your shape to pass through corresponding gates. "
        "Reach the exit before time runs out, as the required gate shape changes periodically."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. "
        "Press space to cycle through shapes (square, circle, triangle) to pass through matching gates."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60-second time limit
    GATE_CYCLE_TIME = 10 * FPS  # Gates change every 10 seconds

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (50, 50, 70)
    COLOR_EXIT = (100, 255, 100)
    COLOR_UI_TEXT = (220, 220, 255)

    # Shapes
    SHAPE_SQUARE = 0
    SHAPE_CIRCLE = 1
    SHAPE_TRIANGLE = 2
    SHAPES = [SHAPE_SQUARE, SHAPE_CIRCLE, SHAPE_TRIANGLE]
    SHAPE_PROPS = {
        SHAPE_SQUARE: {"color": (255, 80, 80), "speed": 3, "name": "SQUARE"},
        SHAPE_CIRCLE: {"color": (80, 120, 255), "speed": 4, "name": "CIRCLE"},
        SHAPE_TRIANGLE: {"color": (255, 255, 80), "speed": 5, "name": "TRIANGLE"},
    }

    # Maze Layout (0: empty, 1: wall, 2: gate, 3: exit, 4: start)
    TILE_SIZE = 40
    MAZE_LAYOUT = [
        "1111111111111111",
        "1400100000001001",
        "1000100000001001",
        "1002111111111001",
        "1000000000010001",
        "1111111210010001",
        "1000000010000021",
        "1002100010000001",
        "1000100000000031",
        "1111111111111111",
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_player = pygame.font.SysFont("Verdana", 14, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_pos = None
        self.player_shape = None
        self.player_size = 18
        self.last_space_held = False
        self.last_dist_to_exit = 0.0

        # Build maze structure from layout
        self._build_maze()

    def _build_maze(self):
        """Parses the MAZE_LAYOUT to create lists of rects for collision."""
        self.walls = []
        self.gates = []
        self.exit_rect = None
        self.start_pos = None
        
        maze_width_tiles = len(self.MAZE_LAYOUT[0])
        maze_height_tiles = len(self.MAZE_LAYOUT)
        offset_x = (self.SCREEN_WIDTH - maze_width_tiles * self.TILE_SIZE) // 2
        offset_y = (self.SCREEN_HEIGHT - maze_height_tiles * self.TILE_SIZE) // 2

        for r, row_str in enumerate(self.MAZE_LAYOUT):
            for c, char in enumerate(row_str):
                x, y = c * self.TILE_SIZE + offset_x, r * self.TILE_SIZE + offset_y
                rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
                if char == '1':
                    self.walls.append(rect)
                elif char == '2':
                    self.gates.append(rect)
                elif char == '3':
                    self.exit_rect = rect
                elif char == '4':
                    self.start_pos = rect.center

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = np.array(self.start_pos, dtype=np.float32)
        self.player_shape = self.SHAPE_SQUARE
        self.last_space_held = False
        self.last_dist_to_exit = self._get_dist_to_exit()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # 1. Handle shape transformation on button press
        if space_held and not self.last_space_held:
            # SFX: Transform sound effect
            self.player_shape = (self.player_shape + 1) % len(self.SHAPES)
        self.last_space_held = space_held

        # 2. Handle movement
        speed = self.SHAPE_PROPS[self.player_shape]["speed"]
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_vec[1] = -1  # Up
        elif movement == 2: move_vec[1] = 1  # Down
        elif movement == 3: move_vec[0] = -1  # Left
        elif movement == 4: move_vec[0] = 1  # Right
        
        if np.any(move_vec):
            new_pos = self.player_pos + move_vec * speed
            player_rect = pygame.Rect(0, 0, self.player_size, self.player_size)
            player_rect.center = tuple(new_pos)

            # 3. Collision Detection
            collided = False
            # Wall collision
            if player_rect.collidelist(self.walls) != -1:
                reward -= 1.0  # Wall bump penalty
                # SFX: Wall thud
                collided = True
            
            # Gate collision
            current_gate_shape = self._get_current_gate_shape()
            gate_idx = player_rect.collidelist(self.gates)
            if gate_idx != -1:
                if self.player_shape == current_gate_shape:
                    old_player_rect = pygame.Rect(0, 0, self.player_size, self.player_size)
                    old_player_rect.center = tuple(self.player_pos)
                    if old_player_rect.collidelist(self.gates) == -1:
                        reward += 1.0  # Gate pass reward
                        self.score += 10
                        # SFX: Gate pass chime
                else:
                    reward -= 1.0  # Wrong gate penalty
                    # SFX: Gate reject buzz
                    collided = True
            
            if not collided:
                self.player_pos = new_pos

        # 4. Reward for distance to exit
        current_dist = self._get_dist_to_exit()
        reward += (self.last_dist_to_exit - current_dist) * 0.01
        self.last_dist_to_exit = current_dist

        # 5. Check termination conditions
        self.steps += 1
        terminated = False
        truncated = False
        
        player_rect = pygame.Rect(0, 0, self.player_size, self.player_size)
        player_rect.center = tuple(self.player_pos)

        if player_rect.colliderect(self.exit_rect):
            terminated = True
            reward += 100.0
            self.score += 1000
            # SFX: Victory fanfare
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            reward -= 10.0
            # SFX: Timeout buzzer

        self.game_over = terminated or truncated
        self.score += reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_current_gate_shape(self):
        return (self.steps // self.GATE_CYCLE_TIME) % len(self.SHAPES)

    def _get_dist_to_exit(self):
        return np.linalg.norm(self.player_pos - np.array(self.exit_rect.center))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        # Draw maze elements
        current_gate_shape = self._get_current_gate_shape()
        gate_color = self.SHAPE_PROPS[current_gate_shape]["color"]

        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
        for gate in self.gates:
            pygame.draw.rect(self.screen, self.COLOR_BG, gate)
            pygame.draw.rect(self.screen, gate_color, gate.inflate(-8, -8), 0, border_radius=4)

        # Draw exit with a subtle pulse
        pulse = abs(math.sin(self.steps * 0.1)) * 10
        exit_color = tuple(np.clip(c + pulse, 0, 255) for c in self.COLOR_EXIT)
        pygame.draw.rect(self.screen, exit_color, self.exit_rect, 0, border_radius=4)
        
        self._draw_player()

    def _draw_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        shape_props = self.SHAPE_PROPS[self.player_shape]
        color = shape_props["color"]
        
        # Glow effect by drawing larger, semi-transparent shapes first
        glow_size = self.player_size * 2.5
        glow_alpha = 60
        
        if self.player_shape == self.SHAPE_SQUARE:
            glow_rect = pygame.Rect(0, 0, glow_size, glow_size)
            glow_rect.center = pos
            pygame.gfxdraw.box(self.screen, glow_rect, color + (glow_alpha,))
            
            player_rect = pygame.Rect(0, 0, self.player_size, self.player_size)
            player_rect.center = pos
            pygame.draw.rect(self.screen, color, player_rect, 0, border_radius=2)
            
        elif self.player_shape == self.SHAPE_CIRCLE:
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(glow_size / 2), color + (glow_alpha,))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(self.player_size / 2), color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(self.player_size / 2), color)

        elif self.player_shape == self.SHAPE_TRIANGLE:
            def get_triangle_points(center, size):
                p1 = (center[0], center[1] - size * 0.58)
                p2 = (center[0] - size * 0.5, center[1] + size * 0.29)
                p3 = (center[0] + size * 0.5, center[1] + size * 0.29)
                return [p1, p2, p3]

            pygame.gfxdraw.filled_polygon(self.screen, get_triangle_points(pos, glow_size), color + (glow_alpha,))
            pygame.gfxdraw.aapolygon(self.screen, get_triangle_points(pos, self.player_size), color)
            pygame.gfxdraw.filled_polygon(self.screen, get_triangle_points(pos, self.player_size), color)

    def _render_ui(self):
        # Time remaining
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = f"TIME {time_left:.1f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 15, 10))

        # Score
        score_text = f"SCORE {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (15, 10))
        
        # Current Gate Type indicator
        current_gate_shape = self._get_current_gate_shape()
        gate_props = self.SHAPE_PROPS[current_gate_shape]
        gate_text = f"GATE REQUIRES: {gate_props['name']}"
        gate_surf = self.font_ui.render(gate_text, True, gate_props['color'])
        self.screen.blit(gate_surf, (self.SCREEN_WIDTH // 2 - gate_surf.get_width() // 2, 10))

        # Player form indicator
        player_props = self.SHAPE_PROPS[self.player_shape]
        player_text = f"{player_props['name']}"
        player_surf = self.font_player.render(player_text, True, (255,255,255))
        text_rect = player_surf.get_rect(center=(self.player_pos[0], self.player_pos[1] - self.player_size - 5))
        self.screen.blit(player_surf, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # To run with display, comment out the os.environ line at the top
    # and re-initialize pygame with display
    try:
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Shape Shifter Maze")
    except pygame.error:
        print("No display available, running headlessly.")
        screen = None

    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4

        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment if a display is available
        if screen:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished! Total reward: {total_reward:.2f}, Score: {info['score']}, Steps: {info['steps']}")
            total_reward = 0
            obs, info = env.reset()

        clock.tick(env.FPS)
        
    env.close()