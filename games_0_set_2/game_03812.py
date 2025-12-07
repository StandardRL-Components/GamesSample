
# Generated: 2025-08-28T00:31:16.486246
# Source Brief: brief_03812.md
# Brief Index: 3812

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Arrows to move cursor. Space to select/jump. Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Minimalist Peg Solitaire. Clear the board to a single peg for the highest score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    COLOR_BG = (20, 30, 70)
    COLOR_HOLE = (40, 50, 90)
    COLOR_PEG = (255, 255, 220)
    COLOR_CURSOR = (0, 255, 255)
    COLOR_SELECTED = (255, 255, 0)
    COLOR_VALID_JUMP = (100, 255, 100)
    COLOR_TEXT = (240, 240, 240)
    
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    BOARD_ROWS, BOARD_COLS = 7, 7
    CELL_SIZE = 50
    PEG_RADIUS = int(CELL_SIZE * 0.35)
    HOLE_RADIUS = int(CELL_SIZE * 0.15)
    BOARD_OFFSET_X = (SCREEN_WIDTH - BOARD_COLS * CELL_SIZE) // 2
    BOARD_OFFSET_Y = (SCREEN_HEIGHT - BOARD_ROWS * CELL_SIZE) // 2
    MAX_STEPS = 1000
    PARTICLE_MAX_LIFETIME = 30

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)

        self.board_layout = np.ones((self.BOARD_ROWS, self.BOARD_COLS), dtype=bool)
        self.board_layout[0:2, 0:2] = False
        self.board_layout[0:2, 5:7] = False
        self.board_layout[5:7, 0:2] = False
        self.board_layout[5:7, 5:7] = False
        self.total_positions = np.sum(self.board_layout)
        
        self.pegs = None
        self.peg_count = 0
        self.cursor_pos = (0, 0)
        self.selected_peg = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        
        self.pegs = self.board_layout.copy()
        center_pos = (self.BOARD_ROWS // 2, self.BOARD_COLS // 2)
        self.pegs[center_pos] = False
        self.peg_count = self.total_positions - 1

        self.cursor_pos = center_pos
        self.selected_peg = None
        
        assert len(self._find_all_valid_jumps()) > 0, "Initial board has no valid moves"
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        jump_executed = False

        if shift_pressed and self.selected_peg is not None:
            self.selected_peg = None

        if movement > 0:
            self._move_cursor(movement)

        if space_pressed:
            selection_reward, jump_executed = self._handle_selection()
            reward += selection_reward
            if jump_executed:
                self.score += selection_reward
        
        self.steps += 1
        terminated = False
        
        if self.peg_count == 1:
            terminated = True
            reward += 100
            self.game_over = True
        elif not jump_executed and len(self._find_all_valid_jumps()) == 0:
            terminated = True
            reward += -10
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _move_cursor(self, direction):
        r, c = self.cursor_pos
        dr, dc = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][direction]
        nr, nc = r + dr, c + dc
        
        if 0 <= nr < self.BOARD_ROWS and 0 <= nc < self.BOARD_COLS and self.board_layout[nr, nc]:
            self.cursor_pos = (nr, nc)

    def _handle_selection(self):
        if self.selected_peg is None:
            if self.pegs[self.cursor_pos]:
                self.selected_peg = self.cursor_pos
                return 0, False
            else:
                return -0.2, False

        else:
            start_pos, end_pos = self.selected_peg, self.cursor_pos
            is_valid, jumped_pos = self._is_valid_jump(start_pos, end_pos)
            
            if is_valid:
                self.pegs[start_pos] = False
                self.pegs[jumped_pos] = False
                self.pegs[end_pos] = True
                self.peg_count -= 1
                self.selected_peg = None
                self._create_particles(jumped_pos, 30) # sfx: jump_pop.wav
                
                jump_reward = 1.0
                if self._is_isolated(end_pos):
                    jump_reward += 5.0
                return jump_reward, True
            else:
                self.selected_peg = None
                return -0.2, False

    def _is_valid_jump(self, start_pos, end_pos):
        if start_pos == end_pos or self.pegs[end_pos]:
            return False, None

        sr, sc = start_pos
        er, ec = end_pos
        
        if (abs(sr - er) == 2 and sc == ec) or (abs(sc - ec) == 2 and sr == er):
            jumped_pos = ((sr + er) // 2, (sc + ec) // 2)
            if self.pegs[jumped_pos]:
                return True, jumped_pos
        
        return False, None

    def _find_all_valid_jumps(self):
        jumps = []
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                if self.pegs[r, c]:
                    for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.BOARD_ROWS and 0 <= nc < self.BOARD_COLS and self.board_layout[nr, nc]:
                           if self._is_valid_jump((r, c), (nr, nc))[0]:
                               jumps.append(((r, c), (nr, nc)))
        return jumps

    def _is_isolated(self, pos):
        r, c = pos
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.BOARD_ROWS and 0 <= nc < self.BOARD_COLS and self.board_layout[nr, nc] and self.pegs[nr, nc]:
                    return False
        return True

    def _grid_to_screen(self, r, c):
        x = self.BOARD_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.BOARD_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        return x, y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                if self.board_layout[r, c]:
                    x, y = self._grid_to_screen(r, c)
                    pygame.gfxdraw.aacircle(self.screen, x, y, self.HOLE_RADIUS, self.COLOR_HOLE)
                    pygame.gfxdraw.filled_circle(self.screen, x, y, self.HOLE_RADIUS, self.COLOR_HOLE)
                    
                    if self.pegs[r, c]:
                        pygame.gfxdraw.aacircle(self.screen, x, y, self.PEG_RADIUS, self.COLOR_PEG)
                        pygame.gfxdraw.filled_circle(self.screen, x, y, self.PEG_RADIUS, self.COLOR_PEG)

        if self.selected_peg is not None:
            r, c = self.selected_peg
            x, y = self._grid_to_screen(r, c)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.PEG_RADIUS + 3, self.COLOR_SELECTED)
            
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.BOARD_ROWS and 0 <= nc < self.BOARD_COLS and self.board_layout[nr, nc]:
                    if self._is_valid_jump(self.selected_peg, (nr, nc))[0]:
                        tx, ty = self._grid_to_screen(nr, nc)
                        pygame.gfxdraw.aacircle(self.screen, tx, ty, self.PEG_RADIUS, self.COLOR_VALID_JUMP)

        cx, cy = self._grid_to_screen(*self.cursor_pos)
        cursor_rect = pygame.Rect(cx - self.CELL_SIZE//2, cy - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=4)
        
        self._update_and_draw_particles()

    def _render_ui(self):
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))
        
        pegs_text = f"Pegs: {self.peg_count}"
        pegs_surf = self.font_main.render(pegs_text, True, self.COLOR_TEXT)
        pegs_rect = pegs_surf.get_rect(topright=(self.SCREEN_WIDTH - 15, 10))
        self.screen.blit(pegs_surf, pegs_rect)
        
        if self.game_over:
            msg = "YOU WIN!" if self.peg_count == 1 else "GAME OVER"
            msg_surf = self.font_main.render(msg, True, self.COLOR_SELECTED)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            bg_rect = msg_rect.inflate(20, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 150))
            self.screen.blit(bg_surf, bg_rect)
            self.screen.blit(msg_surf, msg_rect)

    def _create_particles(self, pos, count):
        px, py = self._grid_to_screen(*pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            dx = math.cos(angle) * speed
            dy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, self.PARTICLE_MAX_LIFETIME + 1)
            radius = self.np_random.uniform(1, 4)
            self.particles.append([px, py, dx, dy, lifetime, radius])

    def _update_and_draw_particles(self):
        for p in self.particles:
            p[0] += p[2]  # x += dx
            p[1] += p[3]  # y += dy
            p[4] -= 1     # lifetime--
            
            alpha = max(0, min(255, int(255 * (p[4] / self.PARTICLE_MAX_LIFETIME))))
            color = (*self.COLOR_PEG, alpha)
            
            radius = int(p[5])
            pos = (int(p[0]), int(p[1]))
            
            if radius > 0:
                particle_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (radius, radius), radius)
                self.screen.blit(particle_surf, (pos[0]-radius, pos[1]-radius))

        self.particles = [p for p in self.particles if p[4] > 0]

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "pegs_remaining": self.peg_count,
            "cursor_pos": self.cursor_pos,
            "selected_peg": self.selected_peg
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc is False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import os
    os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Peg Solitaire Gym Environment")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT): action[2] = 1
                elif event.key == pygame.K_r: obs, info = env.reset()
                elif event.key == pygame.K_q: running = False
        
        if any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Term: {terminated}, Info: {info}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)
        
    env.close()