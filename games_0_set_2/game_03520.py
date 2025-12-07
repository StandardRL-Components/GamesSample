
# Generated: 2025-08-27T23:36:01.454526
# Source Brief: brief_03520.md
# Brief Index: 3520

        
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
        "Controls: Arrow keys to move your green square. Squash bugs by moving next to them."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade action game. Squash all the descending bugs before they reach the bottom or time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 32, 20
    CELL_SIZE = 20
    
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_PLAYER_SAFE = (50, 255, 150)
    COLOR_PLAYER_RISKY = (255, 100, 100)
    COLOR_PLAYER_GLOW = (255, 255, 255, 50)
    COLOR_BUG = (180, 120, 80)
    COLOR_BUG_EYE = (255, 0, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    # Game Parameters
    NUM_BUGS_START = 25
    MAX_LEVELS = 3
    MAX_STEPS_PER_LEVEL = 600
    
    # Rewards
    REWARD_SQUASH = 1.0
    REWARD_SAFE_MOVE_PENALTY = -0.01
    REWARD_LEVEL_CLEAR = 5.0
    REWARD_GAME_WIN = 100.0
    REWARD_LOSS_BUG = -100.0
    REWARD_TIMEOUT = -50.0

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game state variables initialized in reset()
        self.level = 0
        self.total_score = 0
        self.steps_this_episode = 0
        self.steps_remaining_in_level = 0
        self.player_pos = [0, 0]
        self.bugs = []
        self.particles = []
        self.bug_move_accumulator = 0.0
        self.game_over = False
        self.win_message = ""
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.level = 1
        self.total_score = 0
        self.steps_this_episode = 0
        self.steps_remaining_in_level = self.MAX_STEPS_PER_LEVEL
        
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT - 2]
        self.bug_move_accumulator = 0.0
        self.particles = []
        self.game_over = False
        self.win_message = ""

        self._spawn_bugs()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps_this_episode += 1
        self.steps_remaining_in_level -= 1
        
        reward = 0
        terminated = False
        
        # 1. Player Movement
        prev_pos = list(self.player_pos)
        self._move_player(movement)
        
        # 2. Squash Logic
        squashed_bugs_info = self._squash_bugs()
        if squashed_bugs_info:
            num_squashed = len(squashed_bugs_info)
            reward += num_squashed * self.REWARD_SQUASH
            self.total_score += num_squashed
            for pos in squashed_bugs_info:
                self._create_squash_particles(pos)
                # SFX: // Play squash sound
        elif movement != 0: # Penalize non-squashing moves
            reward += self.REWARD_SAFE_MOVE_PENALTY

        # 3. Bug Movement
        loss_occurred = self._move_bugs()
        
        # 4. Update Particles
        self._update_particles()
        
        # 5. Check Termination and Progression
        if loss_occurred:
            reward += self.REWARD_LOSS_BUG
            terminated = True
            self.game_over = True
            self.win_message = "GAME OVER"
            # SFX: // Play game over sound
        elif self.steps_remaining_in_level <= 0:
            reward += self.REWARD_TIMEOUT
            terminated = True
            self.game_over = True
            self.win_message = "TIME OUT"
            # SFX: // Play timeout sound
        elif not self.bugs:
            if self.level < self.MAX_LEVELS:
                reward += self.REWARD_LEVEL_CLEAR
                self._start_next_level()
                # SFX: // Play level clear sound
            else:
                reward += self.REWARD_GAME_WIN
                terminated = True
                self.game_over = True
                self.win_message = "YOU WIN!"
                # SFX: // Play game win fanfare
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _start_next_level(self):
        self.level += 1
        self.steps_remaining_in_level = self.MAX_STEPS_PER_LEVEL
        self.bug_move_accumulator = 0.0
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT - 2]
        self._spawn_bugs()

    def _spawn_bugs(self):
        self.bugs = []
        spawn_area_top = 1
        spawn_area_bottom = self.GRID_HEIGHT // 2
        
        occupied_positions = set()
        
        for _ in range(self.NUM_BUGS_START):
            while True:
                x = self.np_random.integers(0, self.GRID_WIDTH)
                y = self.np_random.integers(spawn_area_top, spawn_area_bottom)
                if (x, y) not in occupied_positions and (x,y) != tuple(self.player_pos):
                    self.bugs.append([x, y])
                    occupied_positions.add((x,y))
                    break
    
    def _move_player(self, movement):
        if movement == 1: # Up
            self.player_pos[1] -= 1
        elif movement == 2: # Down
            self.player_pos[1] += 1
        elif movement == 3: # Left
            self.player_pos[0] -= 1
        elif movement == 4: # Right
            self.player_pos[0] += 1
        
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)

    def _squash_bugs(self):
        px, py = self.player_pos
        neighbors = [(px, py - 1), (px, py + 1), (px - 1, py), (px + 1, py)]
        
        bugs_to_squash_indices = []
        squashed_positions = []
        for i, bug_pos in enumerate(self.bugs):
            if tuple(bug_pos) in neighbors:
                bugs_to_squash_indices.append(i)
                squashed_positions.append(bug_pos)
        
        # Remove squashed bugs by index, in reverse order to avoid index shifting
        for i in sorted(bugs_to_squash_indices, reverse=True):
            del self.bugs[i]
            
        return squashed_positions

    def _move_bugs(self):
        bug_speed = 1.0 - (self.level - 1) * 0.02
        self.bug_move_accumulator += bug_speed
        
        if self.bug_move_accumulator >= 1.0:
            self.bug_move_accumulator -= 1.0
            
            bug_positions_set = {tuple(b) for b in self.bugs}
            
            for bug in self.bugs:
                next_pos = (bug[0], bug[1] + 1)
                # Bugs move down unless another bug is directly below them
                if next_pos not in bug_positions_set:
                    bug[1] += 1
            
            # Check for loss condition
            for bug in self.bugs:
                if bug[1] >= self.GRID_HEIGHT - 1:
                    return True # Loss occurred
        return False # No loss

    def _create_squash_particles(self, pos):
        cx = (pos[0] + 0.5) * self.CELL_SIZE
        cy = (pos[1] + 0.5) * self.CELL_SIZE
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, 20)
            size = self.np_random.integers(3, 6)
            self.particles.append({'pos': [cx, cy], 'vel': vel, 'life': life, 'max_life': life, 'size': size})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_bugs()
        self._render_player()
        self._render_particles()

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        # Highlight bottom row
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_RISKY, (0, (self.GRID_HEIGHT-1)*self.CELL_SIZE, self.SCREEN_WIDTH, self.CELL_SIZE), 2)

    def _render_bugs(self):
        eye_size = self.CELL_SIZE // 5
        for bug in self.bugs:
            x, y = bug
            rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            
            # Pulsing animation
            pulse = (math.sin(self.steps_this_episode * 0.2 + x) + 1) / 2
            inset = int(pulse * 2)
            
            body_rect = rect.inflate(-inset*2, -inset*2)
            pygame.draw.rect(self.screen, self.COLOR_BUG, body_rect, border_radius=4)
            
            # Eyes
            eye_y = body_rect.centery - eye_size // 2
            eye_x1 = body_rect.centerx - self.CELL_SIZE // 4
            eye_x2 = body_rect.centerx + self.CELL_SIZE // 4
            pygame.draw.circle(self.screen, self.COLOR_BUG_EYE, (eye_x1, eye_y), eye_size)
            pygame.draw.circle(self.screen, self.COLOR_BUG_EYE, (eye_x2, eye_y), eye_size)

    def _render_player(self):
        px, py = self.player_pos
        is_risky = any(
            tuple(bug) in [(px, py-1), (px, py+1), (px-1, py), (px+1, py)] for bug in self.bugs
        )
        player_color = self.COLOR_PLAYER_RISKY if is_risky else self.COLOR_PLAYER_SAFE
        
        center_x = int((px + 0.5) * self.CELL_SIZE)
        center_y = int((py + 0.5) * self.CELL_SIZE)

        # Bobbing animation
        bob = math.sin(self.steps_this_episode * 0.3) * 2
        size = self.CELL_SIZE - 4
        
        # Glow effect
        glow_radius = int(size * 0.8)
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, glow_radius, self.COLOR_PLAYER_GLOW)
        
        player_rect = pygame.Rect(0, 0, size, size)
        player_rect.center = (center_x, center_y + bob)
        
        pygame.draw.rect(self.screen, player_color, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, player_rect, width=1, border_radius=3)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = self.COLOR_BUG + (alpha,)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.draw.rect(self.screen, color, (*pos, size, size))

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow_pos = (pos[0] + 2, pos[1] + 2)
            text_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf, shadow_pos)
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Score, Level, Bugs Remaining
        score_text = f"SCORE: {self.total_score}"
        level_text = f"LEVEL: {self.level}/{self.MAX_LEVELS}"
        bugs_text = f"BUGS: {len(self.bugs)}"
        
        draw_text(score_text, self.font_small, self.COLOR_TEXT, (10, 10))
        
        level_surf = self.font_small.render(level_text, True, self.COLOR_TEXT)
        draw_text(level_text, self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH // 2 - level_surf.get_width() // 2, 10))

        bugs_surf = self.font_small.render(bugs_text, True, self.COLOR_TEXT)
        draw_text(bugs_text, self.font_small, self.COLOR_TEXT, (self.SCREEN_WIDTH - bugs_surf.get_width() - 10, 10))
        
        # Timer bar
        timer_percent = self.steps_remaining_in_level / self.MAX_STEPS_PER_LEVEL
        timer_width = (self.SCREEN_WIDTH - 20) * timer_percent
        timer_color = self.COLOR_PLAYER_SAFE if timer_percent > 0.25 else self.COLOR_PLAYER_RISKY
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, 35, self.SCREEN_WIDTH - 20, 10))
        if timer_width > 0:
            pygame.draw.rect(self.screen, timer_color, (10, 35, timer_width, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            text_surf = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.total_score,
            "steps": self.steps_this_episode,
            "level": self.level,
            "bugs_remaining": len(self.bugs)
        }
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    import sys
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Bug Squish")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("      BUG SQUISH DEMO")
    print("="*30)
    print(env.user_guide)
    print("Press R to reset, Q to quit.\n")

    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Only step if a key was pressed, because auto_advance is False
        if action[0] != 0 or env.game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated:
                print(f"Episode Finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                # Wait for reset
        
        # Render the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # We manually control the step rate for human play
        clock.tick(10) # 10 steps per second for human play
        
    pygame.quit()
    sys.exit()