
# Generated: 2025-08-27T17:20:26.879759
# Source Brief: brief_01494.md
# Brief Index: 1494

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Use arrow keys to select a wormhole. Press space to jump. "
        "Reach the blue target wormhole in 10 jumps or less."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a grid-based universe, jumping between unstable wormholes to "
        "reach the target destination within a limited jump budget."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = 35
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.MAX_JUMPS = 10
        self.INITIAL_WORMHOLE_COUNT = 15
        self.WORMHOLE_COLLAPSE_CHANCE = 0.20
        self.MAX_STEPS = 1000

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 36, bold=True)

        # --- Colors ---
        self.COLOR_BG = (10, 10, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150)
        self.COLOR_TARGET = (100, 150, 255)
        self.COLOR_TARGET_GLOW = (100, 150, 255)
        self.COLOR_WORMHOLE = (200, 100, 255)
        self.COLOR_WORMHOLE_GLOW = (200, 100, 255)
        self.COLOR_SELECTION = (255, 255, 0)
        self.COLOR_COLLAPSED = (50, 50, 50)
        self.COLOR_COLLAPSED_GLOW = (80, 80, 80)
        self.COLOR_TRAIL = (200, 220, 255)
        self.COLOR_TARGET_LINE = (100, 150, 255, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_WARN = (255, 50, 50)
        
        # --- Game State (initialized in reset) ---
        self.np_random = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        self.jumps_remaining = 0
        self.player_pos = None
        self.target_pos = None
        self.wormholes = []
        self.collapsed_wormholes = []
        self.selected_wormhole_idx = None # Index in self.wormholes
        self.prev_space_held = False
        self.jump_trails = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        self.jumps_remaining = self.MAX_JUMPS
        self.selected_wormhole_idx = None
        self.prev_space_held = False
        self.jump_trails = []
        self.collapsed_wormholes = []

        all_positions = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(all_positions)

        self.target_pos = all_positions.pop()
        self.player_pos = all_positions.pop()
        
        num_wh = min(len(all_positions), self.INITIAL_WORMHOLE_COUNT)
        self.wormholes = [self.target_pos] + all_positions[:num_wh]

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False

        if not self.game_over:
            if movement > 0:
                self._handle_selection(movement)

            jump_executed = space_held and not self.prev_space_held and self.selected_wormhole_idx is not None
            if jump_executed:
                # SFX: Jump execute sound
                reward = self._execute_jump()
                self.score += reward
                terminated = self.game_over

        self.prev_space_held = space_held
        self.steps += 1
        
        if not terminated and self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_message = "TIME LIMIT REACHED"

        return self._get_observation(), float(reward), terminated, False, self._get_info()

    def _execute_jump(self):
        if self.selected_wormhole_idx is None:
            return 0.0

        # --- Calculate Reward ---
        reward = -0.1  # Base cost for any jump

        start_pos = self.player_pos
        destination_pos = self.wormholes[self.selected_wormhole_idx]
        
        candidate_wormholes = [wh for wh in self.wormholes if wh != start_pos]
        if candidate_wormholes:
            player_screen_pos = np.array(self._grid_to_screen(start_pos))
            distances = [np.linalg.norm(player_screen_pos - np.array(self._grid_to_screen(wh))) for wh in candidate_wormholes]
            avg_dist = np.mean(distances) if distances else 0
            jump_dist = np.linalg.norm(player_screen_pos - np.array(self._grid_to_screen(destination_pos)))
            
            if jump_dist > avg_dist:
                reward += 5.0  # Risky jump bonus
            else:
                reward += -2.0  # Safe jump penalty

        # --- Update State ---
        self.jumps_remaining -= 1
        self.jump_trails.append((start_pos, destination_pos, 25))  # Trail lives for 25 steps
        self.player_pos = destination_pos
        
        # Collapse source wormhole (if not the target)
        if start_pos != self.target_pos and self.np_random.random() < self.WORMHOLE_COLLAPSE_CHANCE:
            if start_pos in self.wormholes:
                self.wormholes.remove(start_pos)
                self.collapsed_wormholes.append(start_pos)
                # SFX: Wormhole collapse sound

        self.selected_wormhole_idx = None

        # --- Check Termination ---
        if self.player_pos == self.target_pos:
            self.game_over = True
            self.win_message = "TARGET REACHED!"
            reward += 100.0  # Win bonus
            # SFX: Victory sound
        elif self.jumps_remaining <= 0:
            self.game_over = True
            self.win_message = "OUT OF JUMPS"
            reward += -50.0  # Loss penalty
            # SFX: Failure sound

        return reward

    def _handle_selection(self, movement):
        # SFX: UI selection tick sound
        direction_vectors = {
            1: np.array([0, -1]), 2: np.array([0, 1]),
            3: np.array([-1, 0]), 4: np.array([1, 0]),
        }
        direction = direction_vectors[movement]
        player_grid_pos = np.array(self.player_pos)
        
        best_dot = -2.0
        best_idx = None

        candidate_indices = [i for i, wh in enumerate(self.wormholes) if wh != self.player_pos]
        if not candidate_indices:
            return

        for i in candidate_indices:
            wh_pos = self.wormholes[i]
            vec_to_wh = np.array(wh_pos) - player_grid_pos
            dist = np.linalg.norm(vec_to_wh)
            if dist < 1e-6: continue
            
            vec_to_wh_normalized = vec_to_wh / dist
            dot = np.dot(direction, vec_to_wh_normalized)
            
            if dot > best_dot and dot > 0.3: # Must be generally in the right quadrant
                best_dot = dot
                best_idx = i
        
        if best_idx is not None:
            self.selected_wormhole_idx = best_idx

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._update_and_draw_trails()
        
        for pos in self.collapsed_wormholes:
            self._render_wormhole(pos, self.COLOR_COLLAPSED, 10, self.COLOR_COLLAPSED_GLOW, is_static=True)

        for i, pos in enumerate(self.wormholes):
            is_selected = (self.selected_wormhole_idx == i)
            if pos == self.target_pos:
                self._render_wormhole(pos, self.COLOR_TARGET, 14, self.COLOR_TARGET_GLOW, is_selected)
            else:
                self._render_wormhole(pos, self.COLOR_WORMHOLE, 10, self.COLOR_WORMHOLE_GLOW, is_selected)
        
        self._render_wormhole(self.player_pos, self.COLOR_PLAYER, 12, self.COLOR_PLAYER_GLOW)

        if not self.game_over:
            p1 = self._grid_to_screen(self.player_pos)
            p2 = self._grid_to_screen(self.target_pos)
            pygame.draw.aaline(self.screen, self.COLOR_TARGET_LINE, p1, p2)

    def _render_ui(self):
        # Score and Jumps
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        jump_color = self.COLOR_TEXT if self.jumps_remaining > 3 else self.COLOR_TEXT_WARN
        jump_text = self.font_ui.render(f"JUMPS: {self.jumps_remaining}", True, jump_color)
        self.screen.blit(jump_text, (self.WIDTH - jump_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            msg_surface = self.font_msg.render(self.win_message, True, self.COLOR_SELECTION)
            msg_rect = msg_surface.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surface, msg_rect)

    def _render_wormhole(self, pos, color, radius, glow_color, is_selected=False, is_static=False):
        screen_pos = self._grid_to_screen(pos)
        
        pulse = 0 if is_static else math.sin(self.steps * 0.15 + pos[0]) * 2
        current_radius = max(1, int(radius + pulse))
        
        glow_radius = int(current_radius * 2.2)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*glow_color, 40), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (screen_pos[0] - glow_radius, screen_pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], current_radius, color)
        pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], current_radius, color)
        
        if is_selected:
            select_radius = int(radius * 1.6)
            angle = (self.steps * 0.1) % (2 * math.pi)
            points = []
            for i in range(3):
                a = angle + i * (2 * math.pi / 3)
                x = screen_pos[0] + select_radius * math.cos(a)
                y = screen_pos[1] + select_radius * math.sin(a)
                points.append((x, y))
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_SELECTION)

    def _grid_to_screen(self, grid_pos):
        x = self.GRID_OFFSET_X + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_OFFSET_Y + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(x), int(y)

    def _draw_grid(self):
        for x in range(self.GRID_SIZE + 1):
            start_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_SIZE * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_SIZE + 1):
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

    def _update_and_draw_trails(self):
        remaining_trails = []
        for start_pos, end_pos, lifetime in self.jump_trails:
            if lifetime > 0:
                p1 = self._grid_to_screen(start_pos)
                p2 = self._grid_to_screen(end_pos)
                alpha = int(255 * (lifetime / 25.0))
                pygame.draw.aaline(self.screen, (*self.COLOR_TRAIL, alpha), p1, p2, 2)
                remaining_trails.append((start_pos, end_pos, lifetime - 1))
        self.jump_trails = remaining_trails

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "jumps_remaining": self.jumps_remaining,
            "player_pos": self.player_pos,
            "target_pos": self.target_pos,
        }

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Wormhole Jumper")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.1f}, Jumps: {info['jumps_remaining']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            
        clock.tick(30) # Limit to 30 FPS for human play
        
    env.close()