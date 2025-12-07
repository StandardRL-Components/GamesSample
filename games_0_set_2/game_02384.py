import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Collect all blue orbs, then find the yellow exit. "
        "Avoid the red ghosts and escape before the timer runs out!"
    )

    game_description = (
        "Navigate a procedurally generated haunted house, collecting spectral orbs while evading ghosts. "
        "Each successful escape makes the next house more dangerous. Clear three houses to win."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 15, 9
        self.CELL_SIZE = 40
        self.X_OFFSET = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2 + 20

        self.MAX_STAGES = 3
        self.TIME_PER_STAGE = 60
        self.ORBS_PER_STAGE = 10

        # --- Colors ---
        self.COLOR_BG = (10, 15, 20)
        self.COLOR_WALL = (40, 45, 50)
        self.COLOR_FLOOR = (20, 25, 30)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255, 50)
        self.COLOR_GHOST = (255, 0, 0)
        self.COLOR_GHOST_GLOW = (255, 50, 50, 100)
        self.COLOR_ORB = (0, 150, 255)
        self.COLOR_ORB_GLOW = (100, 200, 255, 80)
        self.COLOR_EXIT = (255, 255, 0)
        self.COLOR_EXIT_GLOW = (255, 255, 150, 100)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_TIME_BAR_BG = (50, 50, 50)
        self.COLOR_TIME_BAR_FG = (200, 180, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # --- State variables that persist across stages ---
        self.total_episode_score = 0
        self.current_stage = 1
        self.ghosts_to_spawn = 1

        # --- Initialize ---
        # self.reset() is called by the test harness, no need to call it here
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Full reset of the episode
        self.total_episode_score = 0
        self.current_stage = 1
        self.ghosts_to_spawn = 1
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        # Reset stage-specific state
        self.steps = 0
        self.game_over = False
        self.steps_remaining = self.TIME_PER_STAGE
        self.collected_orbs = 0
        
        # Generate maze and place entities
        self.grid = self._generate_maze(self.GRID_WIDTH, self.GRID_HEIGHT)
        floor_tiles_coords = np.argwhere(self.grid == 0)
        self.np_random.shuffle(floor_tiles_coords)
        floor_tiles = [tuple(coord) for coord in floor_tiles_coords]


        # Ensure we have enough floor tiles for all entities
        required_tiles = 1 + self.ghosts_to_spawn + self.ORBS_PER_STAGE + 1
        if len(floor_tiles) < required_tiles:
            # Fallback if maze is too small, though unlikely with current params
            self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.uint8)
            floor_tiles_coords = np.argwhere(self.grid == 0)
            self.np_random.shuffle(floor_tiles_coords)
            floor_tiles = [tuple(coord) for coord in floor_tiles_coords]

        # Player position is (y, x) but we use (x, y) for consistency with pygame coords
        py, px = floor_tiles.pop()
        self.player_pos = (px, py)
        
        self.ghost_pos_list = []
        for _ in range(self.ghosts_to_spawn):
            gy, gx = floor_tiles.pop()
            self.ghost_pos_list.append((gx, gy))

        self.orb_pos_list = []
        for _ in range(self.ORBS_PER_STAGE):
            oy, ox = floor_tiles.pop()
            self.orb_pos_list.append((ox, oy))
        
        ey, ex = floor_tiles.pop()
        self.exit_pos = (ex, ey)

        # Pre-calculate distance for first step's reward
        self.dist_to_nearest_orb = self._get_dist_to_nearest_orb()

    def _generate_maze(self, width, height):
        # Maze represented as 0 for path, 1 for wall
        maze = np.ones((height, width), dtype=np.uint8)
        
        # Use Randomized DFS
        stack = []
        # Note: np.argwhere and maze access use (y, x), but we use (x, y) for positions
        start_y, start_x = self.np_random.integers(0, height), self.np_random.integers(0, width)
        maze[start_y, start_x] = 0
        stack.append((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                nnx, nny = cx + dx*2, cy + dy*2 # Neighbor's neighbor
                if 0 <= nnx < width and 0 <= nny < height and maze[nny, nnx] == 1:
                    neighbors.append((dx, dy))

            if neighbors:
                dx, dy = neighbors[self.np_random.integers(len(neighbors))]
                nx, ny = cx + dx, cy + dy
                nnx, nny = cx + dx*2, cy + dy*2
                maze[ny, nx] = 0
                maze[nny, nnx] = 0
                stack.append((nnx, nny))
            else:
                stack.pop()
        
        return maze

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0.0
        terminated = False

        self.steps += 1
        self.steps_remaining -= 1

        # --- 1. Player Movement and Proximity Reward ---
        old_player_pos = self.player_pos
        px, py = self.player_pos
        
        if movement == 1: py -= 1 # Up
        elif movement == 2: py += 1 # Down
        elif movement == 3: px -= 1 # Left
        elif movement == 4: px += 1 # Right

        if 0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT and self.grid[py, px] == 0:
            self.player_pos = (px, py)

        new_dist = self._get_dist_to_nearest_orb()
        if new_dist < self.dist_to_nearest_orb:
            reward += 1.0 # Closer to an orb
        elif new_dist > self.dist_to_nearest_orb:
            reward -= 0.1 # Further from an orb
        self.dist_to_nearest_orb = new_dist

        # --- 2. Ghost Movement ---
        for i, (gx, gy) in enumerate(self.ghost_pos_list):
            valid_moves = []
            for dx, dy in [(0,0), (0,1), (0,-1), (1,0), (-1,0)]: # Ghosts can stand still
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and self.grid[ny, nx] == 0:
                    valid_moves.append((nx, ny))
            if valid_moves:
                self.ghost_pos_list[i] = valid_moves[self.np_random.integers(len(valid_moves))]

        # --- 3. Check Game Events ---
        # Orb collection
        if self.player_pos in self.orb_pos_list:
            self.orb_pos_list.remove(self.player_pos)
            self.collected_orbs += 1
            reward += 10.0
            self.dist_to_nearest_orb = self._get_dist_to_nearest_orb()

        # Ghost collision (lose)
        if self.player_pos in self.ghost_pos_list:
            reward = -100.0
            self.game_over = True
            terminated = True
        
        # Time out (lose)
        if self.steps_remaining <= 0 and not terminated:
            reward = -100.0
            self.game_over = True
            terminated = True

        # Stage clear (win)
        if self.player_pos == self.exit_pos and self.collected_orbs >= self.ORBS_PER_STAGE and not terminated:
            reward += 100.0
            self.total_episode_score += reward
            
            self.current_stage += 1
            if self.current_stage > self.MAX_STAGES:
                # Episode win
                self.game_over = True
                terminated = True
            else:
                # Go to next stage
                self.ghosts_to_spawn += 1
                self._setup_stage()
                # Return early with new stage observation, reward is from previous stage
                return self._get_observation(), reward, False, False, self._get_info()

        self.total_episode_score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_dist_to_nearest_orb(self):
        if not self.orb_pos_list:
            # If all orbs are collected, measure distance to exit
            return abs(self.player_pos[0] - self.exit_pos[0]) + abs(self.player_pos[1] - self.exit_pos[1])
        
        px, py = self.player_pos
        min_dist = float('inf')
        for ox, oy in self.orb_pos_list:
            dist = abs(px - ox) + abs(py - oy)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw floor and walls
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = (self.X_OFFSET + x * self.CELL_SIZE, self.Y_OFFSET + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                color = self.COLOR_WALL if self.grid[y, x] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)

        # Draw orbs with glow
        for ox, oy in self.orb_pos_list:
            cx = self.X_OFFSET + int((ox + 0.5) * self.CELL_SIZE)
            cy = self.Y_OFFSET + int((oy + 0.5) * self.CELL_SIZE)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, int(self.CELL_SIZE * 0.5), self.COLOR_ORB_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, int(self.CELL_SIZE * 0.25), self.COLOR_ORB)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, int(self.CELL_SIZE * 0.25), self.COLOR_ORB)

        # Draw exit with glow
        ex, ey = self.exit_pos
        exit_rect = (self.X_OFFSET + ex * self.CELL_SIZE, self.Y_OFFSET + ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        if self.collected_orbs >= self.ORBS_PER_STAGE:
            glow_rect = pygame.Rect(exit_rect).inflate(self.CELL_SIZE*0.5, self.CELL_SIZE*0.5)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, self.COLOR_EXIT_GLOW, glow_surf.get_rect(), border_radius=int(self.CELL_SIZE*0.4))
            self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect, border_radius=4)
        
        # Draw ghosts with flicker
        flicker = (math.sin(self.steps * 0.5) + 1) / 2 # Varies between 0 and 1
        glow_alpha = 50 + flicker * 100
        for gx, gy in self.ghost_pos_list:
            cx = self.X_OFFSET + int((gx + 0.5) * self.CELL_SIZE)
            cy = self.Y_OFFSET + int((gy + 0.5) * self.CELL_SIZE)
            size = self.CELL_SIZE * 0.4
            glow_color = (*self.COLOR_GHOST_GLOW[:3], int(glow_alpha))
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, int(size * 1.5), glow_color)
            points = [
                (cx, cy - size),
                (cx - size, cy + size * 0.6),
                (cx + size, cy + size * 0.6)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GHOST)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GHOST)

        # Draw player with glow
        px, py = self.player_pos
        player_rect = (self.X_OFFSET + px * self.CELL_SIZE, self.Y_OFFSET + py * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        glow_rect = pygame.Rect(player_rect).inflate(self.CELL_SIZE*0.25, self.CELL_SIZE*0.25)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=int(self.CELL_SIZE*0.3))
        self.screen.blit(glow_surf, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)

    def _render_ui(self):
        # Time bar
        time_bar_width = self.WIDTH - 20
        time_ratio = max(0, self.steps_remaining / self.TIME_PER_STAGE)
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR_BG, (10, 10, time_bar_width, 15), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR_FG, (10, 10, int(time_bar_width * time_ratio), 15), border_radius=4)

        # Orb count
        orb_text = f"ORBS: {self.collected_orbs} / {self.ORBS_PER_STAGE}"
        self._draw_text(orb_text, (20, self.HEIGHT - 30), self.font_small)

        # Stage count
        stage_text = f"HOUSE: {self.current_stage} / {self.MAX_STAGES}"
        self._draw_text(stage_text, (self.WIDTH - 20, self.HEIGHT - 30), self.font_small, align="right")
        
        # Score
        score_text = f"SCORE: {int(self.total_episode_score)}"
        self._draw_text(score_text, (self.WIDTH // 2, self.HEIGHT - 30), self.font_small, align="center")

    def _draw_text(self, text, pos, font, color=None, align="left"):
        color = color or self.COLOR_UI_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "right":
            text_rect.topright = pos
        elif align == "center":
            text_rect.midtop = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.total_episode_score,
            "steps": self.steps,
            "stage": self.current_stage,
            "orbs_collected": self.collected_orbs,
            "time_remaining": self.steps_remaining,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will create a window and render the game
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False
    
    pygame.display.set_caption("Haunted House Explorer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample()
    action[0] = 0 # No-op
    action[1] = 0 # Released
    action[2] = 0 # Released

    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Human input ---
        movement_action = 0 # No-op by default
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement_action = 1
                elif event.key == pygame.K_DOWN:
                    movement_action = 2
                elif event.key == pygame.K_LEFT:
                    movement_action = 3
                elif event.key == pygame.K_RIGHT:
                    movement_action = 4
                elif event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()
                    print("--- GAME RESET ---")
                elif event.key == pygame.K_q: # Quit
                    done = True

        if movement_action != 0:
            action[0] = movement_action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            
            print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.1f}, Score: {int(info['score'])}, Terminated: {terminated}")

        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(10) # Limit frame rate

        if done:
            print("\n--- GAME OVER ---")
            print(f"Final Score: {int(info['score'])}")
            print("Press 'R' to play again or 'Q' to quit.")
            
            wait_for_input = True
            while wait_for_input:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        wait_for_input = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        done = False
                        wait_for_input = False
                        print("\n--- NEW GAME ---")

    env.close()