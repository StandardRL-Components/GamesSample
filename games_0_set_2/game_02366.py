
# Generated: 2025-08-27T20:09:02.129616
# Source Brief: brief_02366.md
# Brief Index: 2366

        
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
        "Controls: Press space to jump over the scrolling obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-scrolling rhythm game. Jump over neon obstacles to the beat to score points. Miss 5 times and the game is over."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 32)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- Game Constants ---
        self.PLAYER_X = 120
        self.PLAYER_SIZE = 20
        self.GROUND_Y = self.HEIGHT - 80
        self.JUMP_STRENGTH = -11
        self.GRAVITY = 0.5
        self.INITIAL_OBSTACLE_SPEED = 5.0
        self.MAX_MISSES = 5
        self.WIN_OBSTACLES = 100
        self.MAX_STEPS = 30 * 100 # Approx 100 seconds at 30fps
        self.BEAT_PERIOD = 30 # A beat every 30 frames

        # --- Colors ---
        self.COLOR_BG_TOP = (10, 5, 30)
        self.COLOR_BG_BOTTOM = (40, 10, 60)
        self.COLOR_GROUND = (150, 100, 255)
        self.COLOR_PLAYER = (255, 255, 255)
        self.NEON_COLORS = [
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Yellow
            (0, 255, 128)   # Spring Green
        ]
        self.COLOR_PARTICLE = (255, 255, 200)
        self.COLOR_TEXT = (255, 255, 255)

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Player state
        self.player_y = self.GROUND_Y - self.PLAYER_SIZE
        self.player_vy = 0
        self.is_jumping = False
        self.last_space_held = False

        # Game progression
        self.misses = 0
        self.obstacles_cleared = 0
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        
        # Entity lists
        self.obstacles = []
        self.particles = []

        # Timing
        self.beat_counter = 0
        self.frames_since_last_obstacle = 0
        self._spawn_initial_obstacles()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 1.0  # Survival reward

        if not self.game_over:
            # Unpack factorized action
            space_held = action[1] == 1

            # --- Handle Input ---
            if space_held and not self.last_space_held and not self.is_jumping:
                self.is_jumping = True
                self.player_vy = self.JUMP_STRENGTH
                # Sound: Jump

            self.last_space_held = space_held

            # --- Update Game Logic ---
            self._update_player()
            obstacle_reward, collision = self._update_obstacles()
            reward += obstacle_reward
            if collision:
                self.misses += 1
                reward -= 5
                # Sound: Hit/Fail

            self._update_particles()
            self._update_beat()

            # Update score based on cleared obstacles
            self.score = self.obstacles_cleared
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        if self.misses >= self.MAX_MISSES:
            terminated = True
            self.game_over = True
            self.win = False
            reward -= 100
        elif self.obstacles_cleared >= self.WIN_OBSTACLES:
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100
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

    def _update_player(self):
        if self.is_jumping:
            self.player_y += self.player_vy
            self.player_vy += self.GRAVITY
            
            if self.player_y >= self.GROUND_Y - self.PLAYER_SIZE:
                self.player_y = self.GROUND_Y - self.PLAYER_SIZE
                self.player_vy = 0
                self.is_jumping = False
                # Sound: Land

    def _update_obstacles(self):
        reward = 0
        collision_detected = False
        player_rect = pygame.Rect(self.PLAYER_X, self.player_y, self.PLAYER_SIZE, self.PLAYER_SIZE)

        for obstacle in self.obstacles:
            obstacle['x'] -= self.obstacle_speed
            obstacle_rect = pygame.Rect(obstacle['x'], obstacle['y'], obstacle['w'], obstacle['h'])

            # Check for collision
            if not obstacle['cleared'] and player_rect.colliderect(obstacle_rect):
                collision_detected = True
                obstacle['cleared'] = True # Prevent multiple penalties for one obstacle

            # Check for clearing an obstacle
            if not obstacle['cleared'] and obstacle['x'] + obstacle['w'] < self.PLAYER_X:
                obstacle['cleared'] = True
                self.obstacles_cleared += 1
                reward += 10
                self._create_success_particles((self.PLAYER_X, self.player_y + self.PLAYER_SIZE / 2))
                # Sound: Success/Score

                # Increase difficulty
                if self.obstacles_cleared > 0 and self.obstacles_cleared % 20 == 0:
                    self.obstacle_speed += 0.5

        # Remove off-screen obstacles
        self.obstacles = [ob for ob in self.obstacles if ob['x'] > -ob['w']]

        # Spawn new obstacles
        self.frames_since_last_obstacle += 1
        min_dist = max(120, int(self.BEAT_PERIOD * self.obstacle_speed * self.np_random.uniform(0.8, 1.5)))
        if self.frames_since_last_obstacle * self.obstacle_speed > min_dist:
             self._spawn_obstacle()
             self.frames_since_last_obstacle = 0
        
        return reward, collision_detected

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _update_beat(self):
        self.beat_counter = (self.beat_counter + 1) % self.BEAT_PERIOD
    
    def _spawn_initial_obstacles(self):
        x_pos = self.WIDTH
        for _ in range(5):
            x_pos += self.np_random.integers(250, 400)
            self._spawn_obstacle(x_pos=x_pos)

    def _spawn_obstacle(self, x_pos=None):
        if x_pos is None:
            x_pos = self.WIDTH + 50
        
        height = self.np_random.choice([20, 50])
        obstacle = {
            'x': x_pos,
            'y': self.GROUND_Y - height,
            'w': 20,
            'h': height,
            'color': self.np_random.choice(self.NEON_COLORS),
            'cleared': False
        }
        self.obstacles.append(obstacle)

    def _create_success_particles(self, pos):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            particle = {
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'color': self.COLOR_PARTICLE,
                'radius': self.np_random.uniform(1, 3)
            }
            self.particles.append(particle)

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw gradient background
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw ground
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 3)
        for i in range(1, 5):
            alpha = 100 - i * 20
            pygame.draw.line(self.screen, (*self.COLOR_GROUND, alpha), (0, self.GROUND_Y + i*2), (self.WIDTH, self.GROUND_Y + i*2), 2)
        
        # Draw beat indicator
        beat_progress = self.beat_counter / self.BEAT_PERIOD
        pulse = (1 - math.cos(beat_progress * 2 * math.pi)) / 2 # 0 -> 1 -> 0
        beat_radius = int(20 + pulse * 10)
        beat_alpha = int(50 + pulse * 100)
        pygame.gfxdraw.filled_circle(self.screen, self.WIDTH // 2, self.HEIGHT - 30, beat_radius, (*self.COLOR_GROUND, beat_alpha))
        pygame.gfxdraw.aacircle(self.screen, self.WIDTH // 2, self.HEIGHT - 30, beat_radius, (*self.COLOR_GROUND, beat_alpha))

        # Draw obstacles
        for ob in self.obstacles:
            r, g, b = ob['color']
            rect = pygame.Rect(int(ob['x']), int(ob['y']), int(ob['w']), int(ob['h']))
            pygame.draw.rect(self.screen, (r, g, b), rect)
            # Glow effect
            for i in range(3):
                glow_color = (r, g, b, 50 - i * 15)
                pygame.draw.rect(self.screen, glow_color, rect.inflate(i*4, i*4), border_radius=3)

        # Draw player
        player_rect = pygame.Rect(int(self.PLAYER_X), int(self.player_y), self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)
        # Glow effect
        for i in range(4):
            glow_color = (*self.COLOR_PLAYER, 80 - i * 20)
            pygame.draw.rect(self.screen, glow_color, player_rect.inflate(i*3, i*3), border_radius=4)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 20))

        # Misses
        miss_text = "MISSES: " + "X " * self.misses + "- " * (self.MAX_MISSES - self.misses)
        miss_surf = self.font_ui.render(miss_text, True, self.COLOR_TEXT)
        self.screen.blit(miss_surf, (self.WIDTH - miss_surf.get_width() - 20, 20))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            text_surf = self.font_game_over.render(message, True, self.COLOR_PLAYER)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
            "obstacles_cleared": self.obstacles_cleared,
        }

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to run and visualize the game
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for visualization
    pygame.display.set_caption("Rhythm Jumper")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        action = env.action_space.sample() # Start with random action
        action[0] = 0 # No movement
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 0 # No shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward}, Score: {info['score']}, Steps: {info['steps']}")
            # In a real scenario, you might auto-reset. For manual play, we wait for 'r'.
            # obs, info = env.reset()
            # total_reward = 0

        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control frame rate
        clock.tick(30) # Match the BEAT_PERIOD assumption

    env.close()