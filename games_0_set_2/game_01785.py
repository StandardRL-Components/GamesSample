
# Generated: 2025-08-28T02:42:16.580800
# Source Brief: brief_01785.md
# Brief Index: 1785

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Press SPACE to jump to the beat and avoid the red obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced rhythm runner. Jump over procedurally generated obstacles to the beat. "
        "Survive for 60 seconds to win. Three mistakes and you're out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.COURSE_LENGTH_SECONDS = 60
        self.MAX_STEPS = self.COURSE_LENGTH_SECONDS * self.FPS
        self.MAX_MISSES = 3

        # Colors
        self.COLOR_BG_TOP = (43, 2, 89)
        self.COLOR_BG_BOTTOM = (235, 122, 11)
        self.COLOR_GROUND = (20, 10, 30)
        self.COLOR_PLAYER = (0, 191, 255)
        self.COLOR_PLAYER_GLOW = (0, 191, 255, 50)
        self.COLOR_OBSTACLE = (255, 69, 0)
        self.COLOR_BEAT = (50, 255, 50)
        self.COLOR_PARTICLE = (255, 215, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_MISS_X = (200, 0, 0)
        
        # Physics
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15
        self.INITIAL_OBSTACLE_SPEED = 6.0
        self.OBSTACLE_SPEED_INCREASE = 0.25 # per 50 steps

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Pre-render background
        self.background = self._create_gradient_background()

        # Initialize state variables
        self.player_pos = None
        self.player_vel_y = None
        self.on_ground = None
        self.obstacles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.missed_beats = None
        self.game_over = None
        self.obstacle_speed = None
        self.obstacle_spawn_timer = None
        self.beat_timer = None
        
        self.reset()

        # This check is not part of the standard __init__ but is required by the prompt
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.player_pos = [self.WIDTH // 4, self.HEIGHT - 50]
        self.player_vel_y = 0
        self.on_ground = True
        self.obstacles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.missed_beats = 0
        self.game_over = False
        
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.obstacle_spawn_timer = 0
        self.beat_timer = 0

        # Pre-populate some initial obstacles
        for i in range(3):
            self._spawn_obstacle(initial_offset=i * self.WIDTH / 3)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        space_pressed = action[1] == 1
        if space_pressed and self.on_ground:
            self.player_vel_y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump sound

        # --- Game Logic Update ---
        self.steps += 1
        reward = 1.0  # Survival reward

        # Update player
        self._update_player()
        
        # Update obstacles
        self._update_obstacles()

        # Update particles
        self._update_particles()
        
        # Update beat indicator
        self.beat_timer = (self.beat_timer + 1) % 15 # 120 BPM at 30 FPS

        # --- Collision Detection & Scoring ---
        for obstacle in self.obstacles:
            if not obstacle['cleared'] and not obstacle['hit']:
                player_rect = pygame.Rect(self.player_pos[0] - 15, self.player_pos[1] - 15, 30, 30)
                if player_rect.colliderect(obstacle['rect']):
                    self.missed_beats += 1
                    obstacle['hit'] = True
                    reward = -10.0
                    # sfx: collision/fail sound
                    self._spawn_particles(self.player_pos, 20, self.COLOR_OBSTACLE)

            if not obstacle['cleared'] and obstacle['rect'].right < self.player_pos[0]:
                obstacle['cleared'] = True
                if not obstacle['hit']:
                    self.score += 10
                    reward += 5.0
                    # sfx: success chime
                    self._spawn_particles(
                        (obstacle['rect'].centerx, obstacle['rect'].top - 10), 
                        15, 
                        self.COLOR_PARTICLE
                    )

        # --- Termination Check ---
        terminated = False
        if self.missed_beats >= self.MAX_MISSES:
            terminated = True
            reward = -100.0
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            if not self.game_over: # Avoid overwriting loss reward
                reward = 50.0 # Victory reward
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self):
        # Apply gravity
        self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y

        # Ground collision
        ground_level = self.HEIGHT - 50
        if self.player_pos[1] >= ground_level:
            self.player_pos[1] = ground_level
            self.player_vel_y = 0
            if not self.on_ground:
                self.on_ground = True
                # sfx: land sound

    def _update_obstacles(self):
        # Difficulty scaling
        if self.steps > 0 and self.steps % 50 == 0:
            self.obstacle_speed += self.OBSTACLE_SPEED_INCREASE

        # Move existing obstacles
        for obstacle in self.obstacles:
            obstacle['rect'].x -= self.obstacle_speed
        
        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]

        # Spawn new obstacles
        self.obstacle_spawn_timer -= self.obstacle_speed
        if self.obstacle_spawn_timer <= 0:
            self._spawn_obstacle()
            
    def _spawn_obstacle(self, initial_offset=0):
        # Determine spawn timer for next obstacle
        min_dist, max_dist = 250, 450
        self.obstacle_spawn_timer = random.uniform(min_dist, max_dist)
        
        # Determine obstacle type
        is_tall = random.choice([True, False])
        height = 60 if is_tall else 30
        width = 40
        x_pos = self.WIDTH + initial_offset
        y_pos = self.HEIGHT - 50 - height
        
        self.obstacles.append({
            'rect': pygame.Rect(x_pos, y_pos, width, height),
            'cleared': False,
            'hit': False
        })
        
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(15, 30),
                'color': color,
                'radius': random.uniform(1, 4)
            })
    
    def _get_observation(self):
        # Draw background
        self.screen.blit(self.background, (0, 0))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.HEIGHT - 50, self.WIDTH, 50))
        
        # Draw beat indicator
        if self.beat_timer < 3: # Flash for 3 frames
            beat_alpha = 255 - (self.beat_timer * 85)
            s = pygame.Surface((self.WIDTH, 50), pygame.SRCALPHA)
            s.fill((*self.COLOR_BEAT, beat_alpha))
            self.screen.blit(s, (0, self.HEIGHT - 50))
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], p['radius'])
            
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle['rect'])

        # Draw player
        player_x, player_y = int(self.player_pos[0]), int(self.player_pos[1])
        radius = 15
        # Glow effect
        glow_radius = int(radius * 2.5)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (player_x - glow_radius, player_y - glow_radius))
        # Player circle
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, self.COURSE_LENGTH_SECONDS - (self.steps / self.FPS))
        timer_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Missed beats
        miss_text = self.font_large.render("X", True, self.COLOR_MISS_X)
        for i in range(self.missed_beats):
            x_pos = self.WIDTH // 2 - (self.MAX_MISSES * 40) // 2 + i * 40
            self.screen.blit(miss_text, (x_pos, self.HEIGHT - 45))
            
        # Game Over / Win message
        if self.game_over:
            if self.missed_beats >= self.MAX_MISSES:
                msg = "GAME OVER"
            else:
                msg = "COURSE COMPLETE!"
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_gradient_background(self):
        """Creates a pre-rendered surface with the background gradient."""
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            # Interpolate between top and bottom colors
            ratio = y / self.HEIGHT
            r = int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio)
            g = int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio)
            b = int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            pygame.draw.line(bg, (r, g, b), (0, y), (self.WIDTH, y))
        return bg
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "missed_beats": self.missed_beats
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2], f"Action space nvec is {self.action_space.nvec.tolist()}"
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set Pygame to run headlessly if no display is available, or show a window if one is.
    import os
    try:
        os.environ["DISPLAY"]
    except KeyError:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    env.validate_implementation()
    
    # --- Interactive Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Setup for displaying the game window
    if os.environ.get("SDL_VIDEODRIVER") != "dummy":
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Rhythm Runner")
        clock = pygame.time.Clock()
        
        running = True
        while running:
            action = [0, 0, 0] # Default no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                action[1] = 1

            if not done:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            if done:
                # You can add a delay or a "press key to restart" message here
                # For now, we just reset automatically after a short pause
                pygame.time.wait(2000)
                obs, info = env.reset()
                done = False

            # Display the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(env.FPS)
            
        env.close()