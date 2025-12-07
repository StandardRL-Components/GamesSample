
# Generated: 2025-08-28T05:21:01.389694
# Source Brief: brief_05542.md
# Brief Index: 5542

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Escape the maze before the shadows catch you or time runs out."
    )

    game_description = (
        "A tense survival game. Navigate a procedurally generated maze, find the glowing exit, and evade the ever-pursuing shadows. Every step counts."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Configuration ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAZE_W, self.MAZE_H = 15, 11
        self.MAX_STEPS = 60
        self.NUM_SHADOWS = 5

        # --- Colors ---
        self.COLOR_BG = (10, 15, 20)
        self.COLOR_WALL = (50, 60, 70)
        self.COLOR_FLOOR = (20, 25, 30)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_EXIT = (255, 255, 255)
        self.COLOR_EXIT_GLOW = (255, 255, 180)
        self.COLOR_SHADOW = (0, 0, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)

        # --- Isometric Projection ---
        self.tile_w = 28
        self.tile_h = 14
        self.origin_x = self.WIDTH // 2
        self.origin_y = 60
        
        # --- State Variables ---
        self.maze = {}
        self.player_pos = (0, 0)
        self.shadows = []
        self.exit_pos = (0, 0)
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.win = False
        self.loss_reason = ""
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.loss_reason = ""
        self.time_remaining = self.MAX_STEPS

        # Generate maze and entities
        self._generate_maze()
        self.player_pos = (0, 0)
        self.exit_pos = (self.MAZE_W - 1, self.MAZE_H - 1)
        
        # Place shadows
        self.shadows = []
        valid_spawn_points = [
            (x, y) for x in range(self.MAZE_W) for y in range(self.MAZE_H) 
            if self._manhattan_distance((x, y), self.player_pos) > 5 and (x, y) != self.exit_pos
        ]
        if len(valid_spawn_points) < self.NUM_SHADOWS:
            valid_spawn_points = [(x,y) for x in range(self.MAZE_W) for y in range(self.MAZE_H) if (x,y) != self.player_pos and (x,y) != self.exit_pos]

        random.shuffle(valid_spawn_points)
        for i in range(min(self.NUM_SHADOWS, len(valid_spawn_points))):
            self.shadows.append(list(valid_spawn_points[i]))
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held and shift_held are unused per brief
        
        reward = -0.1  # Time penalty
        
        old_player_pos = self.player_pos
        old_dist_to_exit = self._manhattan_distance(old_player_pos, self.exit_pos)
        old_dist_to_shadow = min(self._manhattan_distance(old_player_pos, s) for s in self.shadows) if self.shadows else float('inf')

        # --- Player Movement ---
        new_pos = list(self.player_pos)
        moved = False
        if movement == 1 and 'N' in self.maze.get(self.player_pos, set()): # Up
            new_pos[1] -= 1
            moved = True
        elif movement == 2 and 'S' in self.maze.get(self.player_pos, set()): # Down
            new_pos[1] += 1
            moved = True
        elif movement == 3 and 'W' in self.maze.get(self.player_pos, set()): # Left
            new_pos[0] -= 1
            moved = True
        elif movement == 4 and 'E' in self.maze.get(self.player_pos, set()): # Right
            new_pos[0] += 1
            moved = True
        
        if moved:
            self.player_pos = tuple(new_pos)

        # --- Reward Calculation (based on player move) ---
        new_dist_to_exit = self._manhattan_distance(self.player_pos, self.exit_pos)
        if new_dist_to_exit < old_dist_to_exit:
            reward += 0.2
        
        # Calculate distance to shadows based on player's new position but shadows' old positions
        new_dist_to_shadow = min(self._manhattan_distance(self.player_pos, s) for s in self.shadows) if self.shadows else float('inf')
        if new_dist_to_shadow < old_dist_to_shadow:
            reward -= 1.0

        # --- Shadow Movement ---
        for i, shadow_pos in enumerate(self.shadows):
            self._move_shadow(i)
        
        # --- Update State ---
        self.steps += 1
        self.time_remaining -= 1
        
        # --- Termination Check ---
        terminated = False
        if self.player_pos == self.exit_pos:
            reward += 100.0
            terminated = True
            self.win = True
            self.loss_reason = "ESCAPED!"
        elif any(tuple(s) == self.player_pos for s in self.shadows):
            reward -= 100.0
            terminated = True
            self.loss_reason = "CAUGHT"
        elif self.time_remaining <= 0:
            reward -= 100.0
            terminated = True
            self.loss_reason = "TIME UP"

        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "time_remaining": self.time_remaining,
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
        }

    def _render_game(self):
        # Render floor
        for y in range(self.MAZE_H):
            for x in range(self.MAZE_W):
                px, py = self._grid_to_iso(x, y)
                points = [
                    (px, py),
                    (px + self.tile_w / 2, py + self.tile_h / 2),
                    (px, py + self.tile_h),
                    (px - self.tile_w / 2, py + self.tile_h / 2)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_FLOOR)

        # Render exit
        self._render_glowing_object(self.exit_pos, self.COLOR_EXIT, self.COLOR_EXIT_GLOW, 1.2, 0.6)

        # Render walls
        for y in range(self.MAZE_H):
            for x in range(self.MAZE_W):
                px, py = self._grid_to_iso(x, y)
                passages = self.maze.get((x, y), set())
                if 'S' not in passages:
                    p1 = self._grid_to_iso(x, y + 1)
                    p2 = self._grid_to_iso(x + 1, y + 1)
                    pygame.draw.aaline(self.screen, self.COLOR_WALL, p1, p2, 2)
                if 'E' not in passages:
                    p1 = self._grid_to_iso(x + 1, y)
                    p2 = self._grid_to_iso(x + 1, y + 1)
                    pygame.draw.aaline(self.screen, self.COLOR_WALL, p1, p2, 2)

        # Render shadows
        for shadow_pos in self.shadows:
            self._render_shadow(shadow_pos)

        # Render player
        self._render_glowing_object(self.player_pos, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW, 1.0, 0.4)

    def _render_ui(self):
        # Time remaining
        time_text = self.font_large.render(f"{self.time_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (20, 10))
        
        # Score
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 50))
        
        # Game over message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            color = self.COLOR_EXIT_GLOW if self.win else (255, 80, 80)
            end_text = self.font_large.render(self.loss_reason, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _render_glowing_object(self, pos, color, glow_color, size_mult, glow_mult):
        px, py = self._grid_to_iso(pos[0], pos[1])
        py += self.tile_h / 2
        
        # Glow
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        glow_radius = int((self.tile_w / 3) * size_mult * (1 + pulse * glow_mult))
        
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, (*glow_color, 80))
        pygame.gfxdraw.aacircle(temp_surf, glow_radius, glow_radius, glow_radius, (*glow_color, 80))
        self.screen.blit(temp_surf, (px - glow_radius, py - glow_radius))

        # Main object
        radius = int(self.tile_w / 4 * size_mult)
        pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius, color)
        pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, color)

    def _render_shadow(self, pos):
        px, py = self._grid_to_iso(pos[0], pos[1])
        py += self.tile_h / 2
        
        pulse_x = (math.sin(self.steps * 0.15 + pos[0]) + 1) / 2
        pulse_y = (math.cos(self.steps * 0.15 + pos[1]) + 1) / 2
        
        w = int(self.tile_w * 0.6 * (1 + pulse_x * 0.3))
        h = int(self.tile_h * 0.6 * (1 + pulse_y * 0.3))
        
        shadow_rect = pygame.Rect(px - w, py - h, w * 2, h * 2)
        
        temp_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(temp_surf, (*self.COLOR_SHADOW, 150), (0, 0, *shadow_rect.size))
        self.screen.blit(temp_surf, shadow_rect.topleft)
        # SFX: Shadow pulse sound

    def _generate_maze(self):
        self.maze = {}
        visited = set()
        stack = deque()

        start_pos = (0, 0)
        stack.append(start_pos)
        visited.add(start_pos)
        self.maze[start_pos] = set()

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            
            # N, S, W, E
            moves = [('N', (0, -1)), ('S', (0, 1)), ('W', (-1, 0)), ('E', (1, 0))]
            random.shuffle(moves)

            for direction, (dx, dy) in moves:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.MAZE_W and 0 <= ny < self.MAZE_H and (nx, ny) not in visited:
                    neighbors.append((direction, (nx, ny)))
            
            if neighbors:
                direction, (nx, ny) = neighbors[0]
                
                self.maze.setdefault((cx, cy), set()).add(direction)
                
                opposite = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}[direction]
                self.maze.setdefault((nx, ny), set()).add(opposite)

                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
    
    def _move_shadow(self, shadow_index):
        shadow_pos = self.shadows[shadow_index]
        best_move = list(shadow_pos)
        min_dist = self._manhattan_distance(shadow_pos, self.player_pos)

        moves = [(0, 1, 'S'), (0, -1, 'N'), (1, 0, 'E'), (-1, 0, 'W')]
        random.shuffle(moves)

        for dx, dy, direction in moves:
            if direction in self.maze.get(tuple(shadow_pos), set()):
                next_pos = (shadow_pos[0] + dx, shadow_pos[1] + dy)
                dist = self._manhattan_distance(next_pos, self.player_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_move = list(next_pos)
        
        self.shadows[shadow_index] = best_move
        # SFX: Shadow movement whisper

    def _grid_to_iso(self, x, y):
        iso_x = self.origin_x + (x - y) * (self.tile_w / 2)
        iso_y = self.origin_y + (x + y) * (self.tile_h / 2)
        return int(iso_x), int(iso_y)

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to your actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Maze Escape")
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # No-op, no space, no shift
    
    print(env.user_guide)

    running = True
    while running:
        # --- Human Input ---
        # This part is for human play, not for the Gym environment itself
        movement_action = 0 # Default to no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    movement_action = key_to_action[event.key]
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    done = False
                    action = np.array([0, 0, 0])
                    continue

        # If a movement key was pressed, take a step
        if movement_action != 0:
            action[0] = movement_action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            action[0] = 0 # Reset to no-op for next frame
        
        # --- Rendering ---
        # The environment's observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if done:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            # Wait for reset
            pass
        
        clock.tick(30) # Limit to 30 FPS

    env.close()