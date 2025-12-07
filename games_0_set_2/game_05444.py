import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame


# Set headless mode for pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrows to select a crystal and aim your push. Press space to move it. Match 3+ to score!"
    )

    game_description = (
        "An isometric puzzle game. Strategically push crystals to create matches of 3 or more. Clear the board before you run out of moves!"
    )

    auto_advance = True

    # --- Constants ---
    GRID_SIZE = 10
    NUM_COLORS = 4
    INITIAL_MOVES = 20
    MAX_STEPS = 30 * 60 # 60 seconds at 30fps

    # --- Visuals ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    TILE_WIDTH, TILE_HEIGHT = 48, 24
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = TILE_WIDTH // 2, TILE_HEIGHT // 2
    ORIGIN_X, ORIGIN_Y = SCREEN_WIDTH // 2, 100

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_TEXT = (220, 220, 240)
    COLOR_CURSOR = (100, 255, 255)
    CRYSTAL_COLORS = {
        1: (255, 50, 50),   # Red
        2: (50, 255, 50),   # Green
        3: (50, 100, 255),  # Blue
        4: (255, 255, 50),  # Yellow
    }
    CRYSTAL_GLOW_COLORS = {
        k: tuple(min(255, int(c * 0.5)) for c in v) for k, v in CRYSTAL_COLORS.items()
    }
    PARTICLE_COLORS = list(CRYSTAL_COLORS.values())

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        
        self.game_phase = "INPUT"
        self.animations = deque()
        self.particles = []
        
        # This call is problematic if reset() hangs, which it did.
        # The fix to _generate_initial_board resolves the hang.
        # self.reset() 

    def _iso_to_screen(self, r, c):
        x = self.ORIGIN_X + (c - r) * self.TILE_WIDTH_HALF
        y = self.ORIGIN_Y + (c + r) * self.TILE_HEIGHT_HALF
        return int(x), int(y)

    def _generate_initial_board(self):
        grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        while True:
            matches = self._find_matches(grid)
            if not matches:
                break
            
            # Remove matched gems
            for r_match, c_match in matches:
                grid[r_match, c_match] = 0
            
            # Apply gravity
            for c in range(self.GRID_SIZE):
                empty_r = -1
                for r in range(self.GRID_SIZE - 1, -1, -1):
                    if grid[r, c] == 0 and empty_r == -1:
                        empty_r = r
                    elif grid[r, c] != 0 and empty_r != -1:
                        grid[empty_r, c] = grid[r, c]
                        grid[r, c] = 0
                        empty_r -= 1
            
            # Refill empty cells at the top
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    if grid[r, c] == 0:
                        grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self._generate_initial_board()
        self.visual_grid = [[{'color': self.grid[r][c], 'pos': (r, c), 'scale': 1.0} for c in range(self.GRID_SIZE)] for r in range(self.GRID_SIZE)]

        self.steps = 0
        self.score = 0
        self.moves_left = self.INITIAL_MOVES
        self.game_over = False

        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2], dtype=np.float32)
        self.last_move_dir = np.array([0, 1]) # Default push right
        self.space_was_held = True

        self.game_phase = "INPUT" # INPUT, ANIMATING
        self.animations = deque()
        self.particles = []
        self.step_reward = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        self.step_reward = 0
        terminated = False
        truncated = False

        self._handle_input(movement, space_held)
        self._update_animations()

        # Moved from _render_game to ensure game logic is not in rendering
        if any(a['type'] == 'gravity_check' for a in self.animations):
            self._apply_gravity()

        if not self.animations and self.game_phase == "ANIMATING":
            chain_reaction = self._process_board()
            if not chain_reaction:
                self.game_phase = "INPUT"
                if self.step_reward == 0:
                    self.step_reward = -0.1 # Penalty for a non-matching move

        if self.game_phase == "INPUT" and not self.game_over:
            is_win = np.all(self.grid == 0)
            is_loss = self.moves_left <= 0 and not is_win
            if is_win:
                self.step_reward += 100
                terminated = True
                self.game_over = True
            elif is_loss:
                self.step_reward -= 100
                terminated = True
                self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score = round(self.score + self.step_reward, 2)
        
        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held):
        if self.game_phase != "INPUT":
            return

        move_vec = np.array([0, 0])
        if movement == 1: move_vec = np.array([-1, 0])  # Up
        elif movement == 2: move_vec = np.array([1, 0])   # Down
        elif movement == 3: move_vec = np.array([0, -1])  # Left
        elif movement == 4: move_vec = np.array([0, 1])  # Right
        
        if np.any(move_vec):
            self.last_move_dir = move_vec
            target_pos = self.cursor_pos + move_vec * 0.4
            self.cursor_pos = np.clip(target_pos, 0, self.GRID_SIZE - 1)

        if space_held and not self.space_was_held:
            r, c = int(round(self.cursor_pos[0])), int(round(self.cursor_pos[1]))
            if 0 <= r < self.GRID_SIZE and 0 <= c < self.GRID_SIZE and self.grid[r, c] != 0:
                self.moves_left -= 1
                self.game_phase = "ANIMATING"
                
                dr, dc = self.last_move_dir
                nr, nc = (r + dr), (c + dc)

                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                    self.grid[r, c], self.grid[nr, nc] = self.grid[nr, nc], self.grid[r, c]
                    self.animations.append({'type': 'move', 'r': r, 'c': c, 'target_r': nr, 'target_c': nc, 'progress': 0.0})
                    if self.grid[r,c] != 0:
                        self.animations.append({'type': 'move', 'r': nr, 'c': nc, 'target_r': r, 'target_c': c, 'progress': 0.0})
                else: # Push off edge
                    self.grid[r, c] = 0
                    self.animations.append({'type': 'fade', 'r': r, 'c': c, 'progress': 0.0})
        
        self.space_was_held = space_held

    def _update_animations(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[3] += 0.1 # Gravity
            p[4] -= 1

        if not self.animations:
            return

        anim_speed = 0.15
        for anim in list(self.animations):
            anim['progress'] = min(1.0, anim['progress'] + anim_speed)
            
            if anim['type'] == 'move':
                r, c = anim['r'], anim['c']
                tr, tc = anim['target_r'], anim['target_c']
                self.visual_grid[r][c]['pos'] = (r + (tr - r) * anim['progress'], c + (tc - c) * anim['progress'])
            elif anim['type'] == 'fade':
                self.visual_grid[anim['r']][anim['c']]['scale'] = 1.0 - anim['progress']

            if anim['progress'] >= 1.0:
                self.animations.remove(anim)
                if anim['type'] == 'move':
                    r, c = anim['r'], anim['c']
                    tr, tc = anim['target_r'], anim['target_c']
                    self.visual_grid[r][c], self.visual_grid[tr][tc] = self.visual_grid[tr][tc], self.visual_grid[r][c]
                    self.visual_grid[r][c]['pos'] = (r,c)
                    self.visual_grid[tr][tc]['pos'] = (tr,tc)
                elif anim['type'] == 'fade':
                    self.visual_grid[anim['r']][anim['c']]['color'] = 0

    def _process_board(self):
        matches = self._find_matches(self.grid)
        if not matches:
            return False

        match_groups = self._get_match_groups(matches)
        for group in match_groups:
            self.step_reward += len(group)
            if len(group) >= 4:
                self.step_reward += 5

        for r, c in matches:
            self.animations.append({'type': 'fade', 'r': r, 'c': c, 'progress': 0.0})
            self.grid[r, c] = 0
            screen_x, screen_y = self._iso_to_screen(r, c)
            for _ in range(10):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
                particle_color_idx = self.np_random.integers(len(self.PARTICLE_COLORS))
                particle_color = self.PARTICLE_COLORS[particle_color_idx]
                self.particles.append([screen_x, screen_y, math.cos(angle) * speed, math.sin(angle) * speed, self.np_random.integers(20, 40), particle_color])
        
        self.animations.append({'type': 'gravity_check', 'progress': 0.0})
        return True

    def _apply_gravity(self):
        for anim in self.animations:
            if anim['type'] == 'gravity_check':
                self.animations.remove(anim)
                break
        
        for c in range(self.GRID_SIZE):
            empty_r = -1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] == 0 and empty_r == -1:
                    empty_r = r
                elif self.grid[r, c] != 0 and empty_r != -1:
                    self.grid[empty_r, c] = self.grid[r, c]
                    self.grid[r, c] = 0
                    self.animations.append({'type': 'move', 'r': r, 'c': c, 'target_r': empty_r, 'target_c': c, 'progress': 0.0})
                    empty_r -= 1
        
    def _find_matches(self, grid):
        to_remove = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if grid[r, c] == 0: continue
                color = grid[r,c]
                # Horizontal
                if c < self.GRID_SIZE - 2 and grid[r, c+1] == color and grid[r, c+2] == color:
                    i = 0
                    while c + i < self.GRID_SIZE and grid[r, c+i] == color:
                        to_remove.add((r, c+i))
                        i += 1
                # Vertical
                if r < self.GRID_SIZE - 2 and grid[r+1, c] == color and grid[r+2, c] == color:
                    i = 0
                    while r + i < self.GRID_SIZE and grid[r+i, c] == color:
                        to_remove.add((r+i, c))
                        i += 1
        return to_remove

    def _get_match_groups(self, matches):
        if not matches: return []
        q = deque(list(matches))
        visited = set()
        groups = []
        while q:
            start_node = q.popleft()
            if start_node in visited: continue
            group, group_q = set(), deque([start_node])
            visited.add(start_node)
            while group_q:
                r, c = group_q.popleft()
                group.add((r, c))
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    neighbor = (r+dr, c+dc)
                    if neighbor in matches and neighbor not in visited:
                        visited.add(neighbor)
                        group_q.append(neighbor)
            groups.append(group)
        return groups

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for r in range(self.GRID_SIZE + 1):
            p1, p2 = self._iso_to_screen(r, -0.5), self._iso_to_screen(r, self.GRID_SIZE - 0.5)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for c in range(self.GRID_SIZE + 1):
            p1, p2 = self._iso_to_screen(-0.5, c), self._iso_to_screen(self.GRID_SIZE - 0.5, c)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

        sorted_crystals = sorted([(r,c) for r in range(self.GRID_SIZE) for c in range(self.GRID_SIZE)], key=lambda p: p[0]+p[1])
        for r, c in sorted_crystals:
            v_crystal = self.visual_grid[r][c]
            color_idx = v_crystal['color']
            if color_idx == 0: continue
            
            vr, vc = v_crystal['pos']
            scale = v_crystal['scale']
            if scale <= 0: continue

            sx, sy = self._iso_to_screen(vr, vc)
            
            points = [(sx, sy - self.TILE_HEIGHT_HALF * scale), (sx + self.TILE_WIDTH_HALF * scale, sy), (sx, sy + self.TILE_HEIGHT_HALF * scale), (sx - self.TILE_WIDTH_HALF * scale, sy)]
            points = [(int(px), int(py)) for px, py in points]
            
            glow_points = [(sx, sy - (self.TILE_HEIGHT_HALF + 4) * scale), (sx + (self.TILE_WIDTH_HALF + 4) * scale, sy), (sx, sy + (self.TILE_HEIGHT_HALF + 4) * scale), (sx - (self.TILE_WIDTH_HALF + 4) * scale, sy)]
            glow_points = [(int(px), int(py)) for px, py in glow_points]

            pygame.gfxdraw.aapolygon(self.screen, glow_points, self.CRYSTAL_GLOW_COLORS[color_idx])
            pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.CRYSTAL_GLOW_COLORS[color_idx])
            pygame.gfxdraw.aapolygon(self.screen, points, self.CRYSTAL_COLORS[color_idx])
            pygame.gfxdraw.filled_polygon(self.screen, points, self.CRYSTAL_COLORS[color_idx])

        cursor_r, cursor_c = self.cursor_pos
        sx, sy = self._iso_to_screen(cursor_r, cursor_c)
        cursor_points = [(sx, sy - self.TILE_HEIGHT_HALF), (sx + self.TILE_WIDTH_HALF, sy), (sx, sy + self.TILE_HEIGHT_HALF), (sx - self.TILE_WIDTH_HALF, sy)]
        cursor_points = [(int(px), int(py)) for px, py in cursor_points]
        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, cursor_points, 2)

        for p in self.particles:
            pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), int(p[4] / 10))

    def _render_ui(self):
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        if self.game_over:
            is_win = np.all(self.grid == 0)
            end_text_str = "YOU WIN!" if is_win else "GAME OVER"
            end_text = self.font_main.render(end_text_str, True, self.COLOR_CURSOR)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            pygame.draw.rect(self.screen, self.COLOR_BG, text_rect.inflate(20, 20))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "game_phase": self.game_phase
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == "__main__":
    # To run with display, unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Cavern")
    
    action = env.action_space.sample()
    action.fill(0)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space, shift])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        env.clock.tick(30)

    env.close()