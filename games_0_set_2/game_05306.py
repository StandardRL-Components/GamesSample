
# Generated: 2025-08-28T04:36:40.665702
# Source Brief: brief_05306.md
# Brief Index: 5306

        
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
        "Controls: Press SPACE to jump and SHIFT to dash. Time your actions to the beat!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-scrolling rhythm game. Jump to collect green notes and dash through red obstacles to reach the end with high accuracy."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    # Colors
    COLOR_BG_TOP = (15, 10, 40)
    COLOR_BG_BOTTOM = (5, 0, 20)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_DASH = (255, 255, 255)
    COLOR_NOTE = (50, 255, 50)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_LANE = (40, 30, 80)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE_NOTE = (150, 255, 150)
    COLOR_PARTICLE_OBSTACLE = (255, 150, 150)
    # Player
    PLAYER_WIDTH = 30
    PLAYER_HEIGHT = 40
    PLAYER_X = 100
    GROUND_Y = 320
    GRAVITY = 0.8
    JUMP_STRENGTH = -15
    # Dash
    DASH_DURATION = 10  # frames
    # Game
    MAX_STEPS = 1500
    TOTAL_NOTES_TO_WIN = 100
    BEAT_INTERVAL = 30 # frames between spawns
    # Initial Difficulty
    INITIAL_SCROLL_SPEED = 5.0
    INITIAL_NOTE_SPAWN_PROB = 0.75

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # RNG
        self.np_random = None
        
        # Initialize state variables
        self.reset()

        # Run self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback to a default or existing RNG if no seed is provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        # Player state
        self.player_y = self.GROUND_Y
        self.player_vy = 0
        self.is_grounded = True
        self.is_dashing = False
        self.dash_timer = 0

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Rhythm & Spawning
        self.notes = []
        self.obstacles = []
        self.particles = []
        self.notes_hit = 0
        self.notes_missed = 0
        self.total_notes_spawned = 0
        self.combo = 0
        self.spawn_timer = self.BEAT_INTERVAL

        # Difficulty
        self.scroll_speed = self.INITIAL_SCROLL_SPEED
        self.note_spawn_prob = self.INITIAL_NOTE_SPAWN_PROB

        # Action state
        self.last_space_held = False
        self.last_shift_held = False

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        # movement = action[0]  # 0-4: none/up/down/left/right (unused in this game)
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Handle Input ---
        jump_triggered = space_held and not self.last_space_held
        dash_triggered = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        if jump_triggered and self.is_grounded:
            # sfx: jump.wav
            self.player_vy = self.JUMP_STRENGTH
            self.is_grounded = False
        
        if dash_triggered and not self.is_dashing:
            # sfx: dash.wav
            self.is_dashing = True
            self.dash_timer = self.DASH_DURATION

        # --- Update Game Logic ---
        self.steps += 1
        
        # Update player physics
        self.player_vy += self.GRAVITY
        self.player_y += self.player_vy
        if self.player_y >= self.GROUND_Y:
            self.player_y = self.GROUND_Y
            self.player_vy = 0
            self.is_grounded = True

        # Update dash state
        if self.is_dashing:
            self.dash_timer -= 1
            if self.dash_timer <= 0:
                self.is_dashing = False

        # Update difficulty
        if self.steps > 0 and self.steps % 100 == 0:
            self.scroll_speed += 0.05
            self.note_spawn_prob = min(0.95, self.note_spawn_prob + 0.02)

        # Update spawning
        self.spawn_timer -= 1
        if self.spawn_timer <= 0 and self.total_notes_spawned < self.TOTAL_NOTES_TO_WIN:
            self.spawn_timer = self.BEAT_INTERVAL
            self._spawn_entity()
            
        # Update entities and check collisions
        reward += self._update_notes()
        reward += self._update_obstacles()
        self._update_particles()
        
        # Check termination
        terminated = self._check_termination()
        if terminated:
            accuracy = (self.notes_hit / self.total_notes_spawned) if self.total_notes_spawned > 0 else 0
            miss_rate = (self.notes_missed / self.total_notes_spawned) if self.total_notes_spawned > 0 else 0

            if self.total_notes_spawned >= self.TOTAL_NOTES_TO_WIN:
                if accuracy >= 0.8:
                    reward += 100 # Goal-oriented win reward
                else:
                    reward -= 100 # Goal-oriented lose reward
            elif miss_rate >= 0.2:
                 reward -= 100 # Goal-oriented lose reward

        if self.auto_advance:
            self.clock.tick(30)
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_entity(self):
        self.total_notes_spawned += 1
        spawn_x = self.SCREEN_WIDTH + 50
        
        if self.np_random.random() < self.note_spawn_prob:
            # Spawn a note at one of three heights
            spawn_y = self.np_random.choice([self.GROUND_Y, self.GROUND_Y - 80, self.GROUND_Y - 160])
            self.notes.append(pygame.Rect(spawn_x, spawn_y - 10, 20, 20))
        else:
            # Spawn an obstacle on the ground
            self.obstacles.append(pygame.Rect(spawn_x, self.GROUND_Y - 50, 30, 50))
            
    def _update_notes(self):
        reward = 0
        player_rect = self._get_player_rect()
        
        for note in self.notes[:]:
            note.x -= self.scroll_speed
            if note.colliderect(player_rect):
                # sfx: note_hit.wav
                reward += 1
                self.score += 10
                self.notes_hit += 1
                self.combo += 1
                if self.combo > 0 and self.combo % 10 == 0:
                    reward += 5
                self._create_particles(note.center, self.COLOR_PARTICLE_NOTE, 20)
                self.notes.remove(note)
            elif note.right < 0:
                # sfx: miss.wav
                self.notes_missed += 1
                self.combo = 0
                self.notes.remove(note)
        return reward

    def _update_obstacles(self):
        reward = 0
        player_rect = self._get_player_rect()

        for obstacle in self.obstacles[:]:
            obstacle.x -= self.scroll_speed
            if obstacle.colliderect(player_rect):
                if not self.is_dashing:
                    # sfx: obstacle_hit.wav
                    reward -= 1
                    self.score = max(0, self.score - 20)
                    self.combo = 0
                    self._create_particles(obstacle.center, self.COLOR_PARTICLE_OBSTACLE, 30, is_explosion=True)
                    self.obstacles.remove(obstacle)
            elif obstacle.right < 0:
                self.obstacles.remove(obstacle)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if self.total_notes_spawned > 0:
            miss_rate = self.notes_missed / self.total_notes_spawned
            if miss_rate >= 0.2:
                return True
        if self.total_notes_spawned >= self.TOTAL_NOTES_TO_WIN:
            return True
        return False
        
    def _get_player_rect(self):
        return pygame.Rect(self.PLAYER_X, self.player_y - self.PLAYER_HEIGHT, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a vertical gradient for a stylish background
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))
        
        # Draw subtle lane markers
        pygame.draw.line(self.screen, self.COLOR_LANE, (0, self.GROUND_Y + 1), (self.SCREEN_WIDTH, self.GROUND_Y + 1), 2)
        pygame.draw.line(self.screen, self.COLOR_LANE, (0, self.GROUND_Y - 80), (self.SCREEN_WIDTH, self.GROUND_Y - 80), 1)
        pygame.draw.line(self.screen, self.COLOR_LANE, (0, self.GROUND_Y - 160), (self.SCREEN_WIDTH, self.GROUND_Y - 160), 1)


    def _render_game(self):
        # Render obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.COLOR_OBSTACLE), obstacle, 2)

        # Render notes
        for note in self.notes:
            pygame.gfxdraw.filled_circle(self.screen, note.centerx, note.centery, note.width // 2, self.COLOR_NOTE)
            pygame.gfxdraw.aacircle(self.screen, note.centerx, note.centery, note.width // 2, (255, 255, 255))
        
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['x']) - p['size'], int(p['y']) - p['size']))

        # Render player
        player_rect = self._get_player_rect()
        if self.is_dashing:
            # Draw trail
            for i in range(5):
                alpha = 150 - i * 30
                color = (*self.COLOR_PLAYER_DASH, alpha)
                trail_pos_x = player_rect.x - (i + 1) * 10
                trail_rect = pygame.Rect(trail_pos_x, player_rect.y, player_rect.width, player_rect.height)
                temp_surf = pygame.Surface(trail_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(temp_surf, color, temp_surf.get_rect(), border_radius=4)
                self.screen.blit(temp_surf, trail_rect.topleft)
            # Draw main dashing player
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_DASH, player_rect, border_radius=4)
        else:
            # Draw normal player
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Accuracy
        accuracy = (self.notes_hit / self.total_notes_spawned * 100) if self.total_notes_spawned > 0 else 100
        acc_color = (100, 255, 100) if accuracy >= 80 else ((255, 255, 100) if accuracy >= 60 else (255, 100, 100))
        accuracy_text = self.font_main.render(f"ACC: {accuracy:.1f}%", True, acc_color)
        self.screen.blit(accuracy_text, (self.SCREEN_WIDTH - accuracy_text.get_width() - 10, 10))

        # Combo
        if self.combo > 2:
            combo_text = self.font_main.render(f"x{self.combo}", True, self.COLOR_TEXT)
            self.screen.blit(combo_text, (self.PLAYER_X, self.player_y - self.PLAYER_HEIGHT - 30))

        # Beat indicator
        beat_progress = 1.0 - (self.spawn_timer / self.BEAT_INTERVAL)
        radius = int(15 + 15 * beat_progress)
        alpha = int(150 * (1.0 - beat_progress))
        if alpha > 0:
            pygame.gfxdraw.aacircle(self.screen, self.PLAYER_X + 15, self.GROUND_Y - 20, radius, (*self.COLOR_TEXT, alpha))

    def _get_info(self):
        accuracy = (self.notes_hit / self.total_notes_spawned) if self.total_notes_spawned > 0 else 0
        return {
            "score": self.score,
            "steps": self.steps,
            "notes_hit": self.notes_hit,
            "notes_missed": self.notes_missed,
            "accuracy": accuracy,
            "combo": self.combo,
        }

    def _create_particles(self, pos, color, count, is_explosion=False):
        for _ in range(count):
            if is_explosion:
                angle = self.np_random.random() * 2 * math.pi
                speed = 2 + self.np_random.random() * 4
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
            else:
                vx = -self.scroll_speed + (self.np_random.random() - 0.5) * 2
                vy = (self.np_random.random() - 0.8) * 4
            
            life = 20 + self.np_random.integers(0, 20)
            self.particles.append({
                'x': pos[0], 'y': pos[1], 'vx': vx, 'vy': vy, 'life': life, 'max_life': life,
                'color': color, 'size': self.np_random.integers(1, 4)
            })

    def close(self):
        pygame.quit()

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

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a display for manual play
    pygame.display.set_caption("Rhythm Beat Runner")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Action defaults
        movement = 0
        space_held = 0
        shift_held = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Key state handling for continuous actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render the observation from the environment to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False
            total_reward = 0

    env.close()