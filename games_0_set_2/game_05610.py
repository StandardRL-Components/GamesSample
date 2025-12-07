
# Generated: 2025-08-28T05:32:46.345721
# Source Brief: brief_05610.md
# Brief Index: 5610

        
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


# Particle class for effects
class Particle:
    def __init__(self, x, y, color, size, life, vel_x, vel_y):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life
        self.vel_x = vel_x
        self.vel_y = vel_y

    def update(self):
        self.x += self.vel_x
        self.y += self.vel_y
        self.life -= 1
        # Slow down
        self.vel_x *= 0.95
        self.vel_y *= 0.95
        # Shrink
        self.size = max(0, self.size * (self.life / self.max_life))

    def draw(self, surface):
        if self.life > 0:
            pygame.draw.rect(
                surface, self.color, (int(self.x), int(self.y), int(self.size), int(self.size))
            )

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move and squash the bugs."
    )

    game_description = (
        "Hunt down swarming bugs in a grid-based arcade environment for maximum points."
    )

    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 30, 25)
    COLOR_GRID = (30, 50, 40)
    COLOR_PLAYER = (255, 60, 60)
    COLOR_PLAYER_GLOW = (255, 150, 150)
    BUG_COLORS = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (150, 255, 150)]
    COLOR_TEXT = (220, 220, 220)
    COLOR_HIT_FLASH = (255, 0, 0)
    COLOR_SQUASH_PARTICLE = (255, 255, 255)

    # Screen and Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 15
    CELL_SIZE = 24
    PLAY_AREA_WIDTH = GRID_WIDTH * CELL_SIZE
    PLAY_AREA_HEIGHT = GRID_HEIGHT * CELL_SIZE
    OFFSET_X = (SCREEN_WIDTH - PLAY_AREA_WIDTH) // 2
    OFFSET_Y = (SCREEN_HEIGHT - PLAY_AREA_HEIGHT) // 2

    # Game Mechanics
    MAX_STEPS = 2000
    INITIAL_LIVES = 3
    BUGS_TO_WIN = 25
    INITIAL_BUG_COUNT = 5
    INITIAL_BUG_MOVE_PROB = 0.2
    DIFFICULTY_INTERVAL = 5 # Increase difficulty every 5 squashes
    DIFFICULTY_INCREASE = 0.02
    COMBO_DURATION = 90 # steps

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_combo = pygame.font.Font(None, 32)

        # Initialize state variables (will be properly set in reset)
        self.player_pos = [0, 0]
        self.bugs = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.bugs_squashed_total = 0
        self.bugs_to_win_remaining = 0
        self.game_over = False
        self.bug_move_prob = 0.0
        self.difficulty_level = 0
        self.combo = 0
        self.combo_timer = 0
        self.hit_flash_timer = 0

        self.reset()
        
        # This will run a self-check to ensure the implementation matches the spec
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.bugs = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.bugs_squashed_total = 0
        self.bugs_to_win_remaining = self.BUGS_TO_WIN
        self.game_over = False

        self.bug_move_prob = self.INITIAL_BUG_MOVE_PROB
        self.difficulty_level = 0
        
        self.combo = 0
        self.combo_timer = 0
        self.hit_flash_timer = 0

        for _ in range(self.INITIAL_BUG_COUNT):
            self._spawn_bug()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        # --- Update timers ---
        self.steps += 1
        if self.hit_flash_timer > 0:
            self.hit_flash_timer -= 1
        if self.combo_timer > 0:
            self.combo_timer -= 1
        else:
            self.combo = 0

        # --- Calculate reward for movement ---
        nearest_bug_dist_before = self._get_nearest_bug_dist()

        # --- Player Movement ---
        prev_player_pos = list(self.player_pos)
        if movement == 1: # Up
            self.player_pos[1] -= 1
        elif movement == 2: # Down
            self.player_pos[1] += 1
        elif movement == 3: # Left
            self.player_pos[0] -= 1
        elif movement == 4: # Right
            self.player_pos[0] += 1
        
        # Clamp player position to grid
        self.player_pos[0] = max(0, min(self.GRID_WIDTH - 1, self.player_pos[0]))
        self.player_pos[1] = max(0, min(self.GRID_HEIGHT - 1, self.player_pos[1]))
        
        moved = prev_player_pos != self.player_pos
        
        if moved:
            nearest_bug_dist_after = self._get_nearest_bug_dist()
            if nearest_bug_dist_after < nearest_bug_dist_before:
                reward += 1.0
            else:
                reward -= 0.2
        else: # No-op or bumped into wall
            reward -= 0.1

        # --- Bug Movement and Collision ---
        bugs_to_remove = []
        player_hit = False
        
        # Move bugs first
        for bug in self.bugs:
            if self.np_random.random() < self.bug_move_prob:
                move_dir = self.np_random.integers(0, 4)
                if move_dir == 0: bug['pos'][0] += 1 # Right
                elif move_dir == 1: bug['pos'][0] -= 1 # Left
                elif move_dir == 2: bug['pos'][1] += 1 # Down
                elif move_dir == 3: bug['pos'][1] -= 1 # Up
                
                bug['pos'][0] = max(0, min(self.GRID_WIDTH - 1, bug['pos'][0]))
                bug['pos'][1] = max(0, min(self.GRID_HEIGHT - 1, bug['pos'][1]))
        
        # Check for collisions after all movement
        for i, bug in enumerate(self.bugs):
            if bug['pos'] == self.player_pos:
                # Two cases: player moved onto bug, or bug moved onto player
                if moved: # Player squashed bug
                    bugs_to_remove.append(i)
                    self.score += 10
                    reward += 10
                    self.combo += 1
                    self.combo_timer = self.COMBO_DURATION
                    combo_bonus = 5 * (self.combo -1)
                    self.score += combo_bonus
                    reward += combo_bonus
                    self.bugs_squashed_total += 1
                    self.bugs_to_win_remaining -= 1
                    
                    # Create squash particles
                    px, py = self._grid_to_pixel(bug['pos'])
                    for _ in range(15):
                        # SFX: SQUISH
                        angle = self.np_random.uniform(0, 2 * math.pi)
                        speed = self.np_random.uniform(2, 5)
                        vel_x = math.cos(angle) * speed
                        vel_y = math.sin(angle) * speed
                        self.particles.append(Particle(px + self.CELL_SIZE/2, py + self.CELL_SIZE/2, self.COLOR_SQUASH_PARTICLE, 6, 20, vel_x, vel_y))

                else: # Bug moved onto player
                    player_hit = True

        if player_hit:
            # SFX: PLAYER_HIT
            self.lives -= 1
            self.score -= 25 # Penalty
            reward -= 50
            self.hit_flash_timer = 5
            self.combo = 0
            self.combo_timer = 0
        
        # Remove squashed bugs and respawn
        if bugs_to_remove:
            # Sort in reverse to avoid index errors
            for i in sorted(bugs_to_remove, reverse=True):
                del self.bugs[i]
            for _ in range(len(bugs_to_remove)):
                if self.bugs_to_win_remaining > 0:
                    self._spawn_bug()
        
        # --- Update Difficulty ---
        new_difficulty_level = self.bugs_squashed_total // self.DIFFICULTY_INTERVAL
        if new_difficulty_level > self.difficulty_level:
            self.difficulty_level = new_difficulty_level
            self.bug_move_prob = min(0.8, self.INITIAL_BUG_MOVE_PROB + self.difficulty_level * self.DIFFICULTY_INCREASE)

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

        # --- Check Termination ---
        terminated = False
        if self.bugs_to_win_remaining <= 0:
            terminated = True
            self.game_over = True
            reward += 100 # Win bonus
        elif self.lives <= 0:
            terminated = True
            self.game_over = True
            reward -= 100 # Loss penalty
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            
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
            "lives": self.lives,
            "bugs_remaining": self.bugs_to_win_remaining,
            "combo": self.combo
        }
        
    def _grid_to_pixel(self, grid_pos):
        x = self.OFFSET_X + grid_pos[0] * self.CELL_SIZE
        y = self.OFFSET_Y + grid_pos[1] * self.CELL_SIZE
        return x, y

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.OFFSET_X + x * self.CELL_SIZE, self.OFFSET_Y)
            end_pos = (self.OFFSET_X + x * self.CELL_SIZE, self.OFFSET_Y + self.PLAY_AREA_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.OFFSET_X, self.OFFSET_Y + y * self.CELL_SIZE)
            end_pos = (self.OFFSET_X + self.PLAY_AREA_WIDTH, self.OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

        # Draw bugs
        for bug in self.bugs:
            px, py = self._grid_to_pixel(bug['pos'])
            jitter_x = self.np_random.uniform(-1, 1)
            jitter_y = self.np_random.uniform(-1, 1)
            bug_rect = pygame.Rect(px + 2 + jitter_x, py + 2 + jitter_y, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
            pygame.draw.rect(self.screen, bug['color'], bug_rect, border_radius=4)
            
        # Draw player
        px, py = self_grid_to_pixel = self._grid_to_pixel(self.player_pos)
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 * 4  # 0 to 4
        
        # Glow effect
        glow_size = self.CELL_SIZE + pulse
        glow_rect = pygame.Rect(
            px - (glow_size - self.CELL_SIZE) / 2, 
            py - (glow_size - self.CELL_SIZE) / 2, 
            glow_size, 
            glow_size
        )
        s = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_PLAYER_GLOW, 50), s.get_rect(), border_radius=int(glow_size/3))
        self.screen.blit(s, glow_rect.topleft)

        player_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Draw hit flash
        if self.hit_flash_timer > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = 150 * (self.hit_flash_timer / 5)
            flash_surface.fill((*self.COLOR_HIT_FLASH, alpha))
            self.screen.blit(flash_surface, (0, 0))
            
    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 15))

        # Bugs to win
        bugs_text = self.font_small.render(f"TARGET: {self.bugs_to_win_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(bugs_text, (self.SCREEN_WIDTH - bugs_text.get_width() - 15, 15))

        # Lives
        life_icon_size = 16
        life_spacing = 6
        total_lives_width = self.lives * life_icon_size + max(0, self.lives - 1) * life_spacing
        start_x = (self.SCREEN_WIDTH - total_lives_width) // 2
        for i in range(self.lives):
            rect = (start_x + i * (life_icon_size + life_spacing), 15, life_icon_size, life_icon_size)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=3)
            
        # Combo
        if self.combo > 1:
            combo_text = self.font_combo.render(f"x{self.combo}", True, self.BUG_COLORS[self.combo % len(self.BUG_COLORS)])
            player_px, player_py = self._grid_to_pixel(self.player_pos)
            text_x = player_px + self.CELL_SIZE
            text_y = player_py - self.CELL_SIZE
            self.screen.blit(combo_text, (text_x, text_y))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.bugs_to_win_remaining <= 0:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
    def _spawn_bug(self):
        while True:
            pos = [
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            ]
            # Ensure not spawning on player or other bugs
            if pos != self.player_pos and not any(bug['pos'] == pos for bug in self.bugs):
                color_index = self.np_random.integers(0, len(self.BUG_COLORS))
                self.bugs.append({'pos': pos, 'color': self.BUG_COLORS[color_index]})
                break
                
    def _get_nearest_bug_dist(self):
        if not self.bugs:
            return float('inf')
        
        player_p = np.array(self.player_pos)
        bug_positions = np.array([bug['pos'] for bug in self.bugs])
        distances = np.linalg.norm(bug_positions - player_p, axis=1)
        return np.min(distances)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up a window to see the game
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bug Squasher")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        movement = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        elif keys[pygame.K_r]: # Reset game
             obs, info = env.reset()
             total_reward = 0
             done = False
        
        # The action is a list [movement, space, shift]
        action = [movement, 0, 0] # Space and shift are not used in this game
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # The brief specifies frame-by-frame state, so we wait for input
        # We add a small delay to make it playable by humans
        clock.tick(10) # Human play speed, 10 actions per second

    env.close()