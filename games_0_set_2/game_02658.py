
# Generated: 2025-08-27T21:01:52.922708
# Source Brief: brief_02658.md
# Brief Index: 2658

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to navigate the maze. "
        "Collect all the yellow gems and avoid the red mines."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Navigate a procedurally generated isometric maze, "
        "collecting gems while avoiding mines to achieve the highest score before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAZE_WIDTH = 15
    MAZE_HEIGHT = 15
    NUM_GEMS = 25
    NUM_MINES = 15
    MAX_STEPS = 1800 # Step limit, equivalent to 60s @ 30fps if auto-advancing

    # Colors
    COLOR_BG = (50, 50, 60)
    COLOR_WALL_TOP = (100, 100, 110)
    COLOR_WALL_SIDE = (80, 80, 90)
    COLOR_FLOOR = (70, 70, 80)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (100, 200, 255)
    COLOR_GEM = (255, 220, 0)
    COLOR_GEM_GLOW = (255, 255, 150)
    COLOR_MINE = (255, 50, 50)
    COLOR_MINE_GLOW = (255, 150, 150)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)

    # Isometric projection
    TILE_WIDTH_HALF = 20
    TILE_HEIGHT_HALF = 10
    ISO_ORIGIN_X = SCREEN_WIDTH // 2
    ISO_ORIGIN_Y = 60

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # Internal state variables are initialized in reset()
        self.player_pos = None
        self.gems = None
        self.mines = None
        self.maze = None
        self.floor_tiles = None
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.np_random = None

        # This call is for development; it ensures the implementation is correct.
        self._initialize_state_for_validation()
        self.validate_implementation()

    def _initialize_state_for_validation(self):
        """A minimal setup to pass validation. reset() will do the full setup."""
        self.np_random, _ = gym.utils.seeding.np_random()
        self.maze = {(x, y): {'N': 1, 'S': 1, 'W': 1, 'E': 1} for x in range(self.MAZE_WIDTH) for y in range(self.MAZE_HEIGHT)}
        self.player_pos = (0, 0)
        self.gems = []
        self.mines = []
        self.floor_tiles = []
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        
        self._generate_maze()
        self._place_objects()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        
        # --- Reward Calculation ---
        # 1. Proximity-based reward (calculated based on the move)
        prev_pos = self.player_pos
        
        # --- Game Logic ---
        # 2. Player Movement
        next_pos = list(self.player_pos)
        if movement == 1 and self.maze[self.player_pos]['N'] == 0: # Up (Iso Up-Left)
            next_pos[1] -= 1
        elif movement == 2 and self.maze[self.player_pos]['S'] == 0: # Down (Iso Down-Right)
            next_pos[1] += 1
        elif movement == 3 and self.maze[self.player_pos]['W'] == 0: # Left (Iso Down-Left)
            next_pos[0] -= 1
        elif movement == 4 and self.maze[self.player_pos]['E'] == 0: # Right (Iso Up-Right)
            next_pos[0] += 1
        
        self.player_pos = tuple(next_pos)
        
        reward = self._calculate_proximity_reward(prev_pos, self.player_pos)
        if movement != 0:
            pass # sfx: player_move

        # 3. Event-based rewards & state changes
        if self.player_pos in self.gems:
            # sfx: gem_collect
            self.gems.remove(self.player_pos)
            self.gems_collected += 1
            reward += 10
            self.score += 100
        
        if self.player_pos in self.mines:
            # sfx: mine_explosion
            self.game_over = True
            reward -= 100
            self.score -= 500

        # --- Termination Check ---
        terminated = False
        if self.game_over:
            terminated = True
        elif self.gems_collected == self.NUM_GEMS:
            # sfx: level_complete
            terminated = True
            self.game_over = True
            reward += 50
            self.score += 1000
        elif self.steps >= self.MAX_STEPS:
            # sfx: time_up
            terminated = True
            self.game_over = True
            self.score -= 200

        self.score += reward # Add proximity reward to score for display
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _calculate_proximity_reward(self, old_pos, new_pos):
        if not self.gems: return 0
        
        def get_dist(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        # Gem proximity
        old_gem_dist = min(get_dist(old_pos, g) for g in self.gems)
        new_gem_dist = min(get_dist(new_pos, g) for g in self.gems)
        gem_reward = (old_gem_dist - new_gem_dist) * 0.1

        # Mine proximity
        if not self.mines:
            mine_reward = 0
        else:
            old_mine_dist = min(get_dist(old_pos, m) for m in self.mines)
            new_mine_dist = min(get_dist(new_pos, m) for m in self.mines)
            mine_reward = (new_mine_dist - old_mine_dist) * 0.2 # Positive if moving away

        return gem_reward - mine_reward

    def _generate_maze(self):
        self.maze = {(x, y): {'N': 1, 'S': 1, 'W': 1, 'E': 1} for x in range(self.MAZE_WIDTH) for y in range(self.MAZE_HEIGHT)}
        stack = []
        visited = set()
        
        start_x, start_y = self.np_random.integers(0, self.MAZE_WIDTH), self.np_random.integers(0, self.MAZE_HEIGHT)
        stack.append((start_x, start_y))
        visited.add((start_x, start_y))

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            if cy > 0 and (cx, cy - 1) not in visited: neighbors.append(('N', (cx, cy - 1)))
            if cy < self.MAZE_HEIGHT - 1 and (cx, cy + 1) not in visited: neighbors.append(('S', (cx, cy + 1)))
            if cx > 0 and (cx - 1, cy) not in visited: neighbors.append(('W', (cx - 1, cy)))
            if cx < self.MAZE_WIDTH - 1 and (cx + 1, cy) not in visited: neighbors.append(('E', (cx + 1, cy)))

            if neighbors:
                direction, (nx, ny) = random.choice(neighbors)
                if direction == 'N': self.maze[(cx, cy)]['N'], self.maze[(nx, ny)]['S'] = 0, 0
                elif direction == 'S': self.maze[(cx, cy)]['S'], self.maze[(nx, ny)]['N'] = 0, 0
                elif direction == 'W': self.maze[(cx, cy)]['W'], self.maze[(nx, ny)]['E'] = 0, 0
                elif direction == 'E': self.maze[(cx, cy)]['E'], self.maze[(nx, ny)]['W'] = 0, 0
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

    def _place_objects(self):
        q = [(0, 0)]; reachable = {(0, 0)}; head = 0
        while head < len(q):
            x, y = q[head]; head += 1
            if self.maze[(x,y)]['N'] == 0 and (x, y-1) not in reachable: q.append((x, y-1)); reachable.add((x, y-1))
            if self.maze[(x,y)]['S'] == 0 and (x, y+1) not in reachable: q.append((x, y+1)); reachable.add((x, y+1))
            if self.maze[(x,y)]['W'] == 0 and (x-1, y) not in reachable: q.append((x-1, y)); reachable.add((x-1, y))
            if self.maze[(x,y)]['E'] == 0 and (x+1, y) not in reachable: q.append((x+1, y)); reachable.add((x+1, y))
        
        self.floor_tiles = list(reachable)
        if len(self.floor_tiles) < self.NUM_GEMS + self.NUM_MINES + 1: return self.reset()

        self.player_pos = self.floor_tiles.pop(self.np_random.integers(len(self.floor_tiles)))
        gem_indices = self.np_random.choice(len(self.floor_tiles), self.NUM_GEMS, replace=False)
        self.gems = [self.floor_tiles[i] for i in gem_indices]
        available_for_mines = [tile for i, tile in enumerate(self.floor_tiles) if i not in gem_indices]
        
        if len(available_for_mines) >= self.NUM_MINES:
            mine_indices = self.np_random.choice(len(available_for_mines), self.NUM_MINES, replace=False)
            self.mines = [available_for_mines[i] for i in mine_indices]
        else: self.mines = available_for_mines[:]

    def _iso_to_cart(self, x, y):
        screen_x = self.ISO_ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ISO_ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, x, y, color_top, color_side):
        px, py = self._iso_to_cart(x, y)
        points = [
            (px, py - self.TILE_HEIGHT_HALF), (px + self.TILE_WIDTH_HALF, py),
            (px, py + self.TILE_HEIGHT_HALF), (px - self.TILE_WIDTH_HALF, py)
        ]
        pygame.draw.polygon(surface, color_top, points)

    def _render_text(self, surface, text, font, x, y, color, shadow_color, center=False):
        text_surf_s = font.render(text, True, shadow_color)
        text_surf = font.render(text, True, color)
        if center:
            rect_s = text_surf_s.get_rect(center=(x + 2, y + 2))
            rect = text_surf.get_rect(center=(x, y))
        else:
            rect_s, rect = (x + 2, y + 2), (x, y)
        surface.blit(text_surf_s, rect_s)
        surface.blit(text_surf, rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        draw_list = []
        for x in range(self.MAZE_WIDTH):
            for y in range(self.MAZE_HEIGHT):
                draw_list.append(('floor', (x, y)))
        for gem_pos in self.gems: draw_list.append(('gem', gem_pos))
        for mine_pos in self.mines: draw_list.append(('mine', mine_pos))
        draw_list.append(('player', self.player_pos))
        draw_list.sort(key=lambda item: (item[1][0] + item[1][1], item[1][1], item[1][0]))

        for item_type, pos in draw_list:
            px, py = self._iso_to_cart(pos[0], pos[1])
            if item_type == 'floor': self._draw_iso_cube(self.screen, pos[0], pos[1], self.COLOR_FLOOR, self.COLOR_WALL_SIDE)
            elif item_type == 'gem':
                glow_radius = 8 + 2 * math.sin(self.steps * 0.2)
                pygame.gfxdraw.filled_circle(self.screen, px, py, int(glow_radius), self.COLOR_GEM_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, px, py, 6, self.COLOR_GEM)
            elif item_type == 'mine':
                glow_radius = 8 + 3 * math.sin(self.steps * 0.3)
                pygame.gfxdraw.filled_circle(self.screen, px, py, int(glow_radius), self.COLOR_MINE_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, px, py, 5, self.COLOR_MINE)
            elif item_type == 'player':
                pygame.gfxdraw.filled_circle(self.screen, px, py, 9, self.COLOR_PLAYER_GLOW)
                pygame.draw.rect(self.screen, self.COLOR_PLAYER, pygame.Rect(px - 5, py - 5, 10, 10))
        
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                p1 = self._iso_to_cart(x, y)
                if self.maze[(x,y)]['N'] == 1: pygame.draw.line(self.screen, self.COLOR_WALL_TOP, p1, self._iso_to_cart(x + 1, y), 3)
                if self.maze[(x,y)]['W'] == 1: pygame.draw.line(self.screen, self.COLOR_WALL_TOP, p1, self._iso_to_cart(x, y + 1), 3)

    def _render_ui(self):
        gem_text = f"Gems: {self.gems_collected} / {self.NUM_GEMS}"
        self._render_text(self.screen, gem_text, self.font_medium, 10, 10, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        time_left = max(0, self.MAX_STEPS - self.steps)
        timer_text = f"Steps: {time_left}"
        self._render_text(self.screen, timer_text, self.font_medium, self.SCREEN_WIDTH - 160, 10, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        score_text = f"Score: {int(self.score)}"
        self._render_text(self.screen, score_text, self.font_large, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 45, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, center=True)
        
        if self.game_over:
            msg = ""
            if self.gems_collected == self.NUM_GEMS: msg = "LEVEL COMPLETE!"
            elif self.player_pos in self.mines: msg = "GAME OVER - MINE!"
            elif self.steps >= self.MAX_STEPS: msg = "GAME OVER - TIME UP!"
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            self._render_text(self.screen, msg, self.font_large, self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW, center=True)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    running = True
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Runner")
    clock = pygame.time.Clock()
    action = [0, 0, 0]
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while running:
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if event.type == pygame.KEYDOWN:
                current_action = [0, 0, 0] 
                if event.key == pygame.K_UP: current_action[0] = 1
                elif event.key == pygame.K_DOWN: current_action[0] = 2
                elif event.key == pygame.K_LEFT: current_action[0] = 3
                elif event.key == pygame.K_RIGHT: current_action[0] = 4
                if current_action[0] != 0:
                    obs, reward, terminated, truncated, info = env.step(current_action)
                    action_taken = True
                if terminated:
                    print(f"Game Over! Final Score: {info['score']}")
                    action_taken = True # To redraw the final frame
                    pygame.time.wait(2000)
                    obs, info = env.reset()

        # Since auto_advance is False, we only redraw when an action happens or initially
        if action_taken or 'surf' not in locals():
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
        
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    env.close()