
# Generated: 2025-08-28T00:38:25.233099
# Source Brief: brief_03854.md
# Brief Index: 3854

        
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
        "Controls: Press Space to jump on the beat. Timing is everything!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced rhythm runner. Jump over neon obstacles by timing your "
        "actions to the pulsating beat to maximize your score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500  # Increased to allow for a longer level
    LEVEL_END_STEP = 1200

    COLOR_BG_TOP = (10, 5, 30)
    COLOR_BG_BOTTOM = (30, 10, 50)
    COLOR_GROUND = (150, 150, 200)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 100, 128)
    COLOR_OBSTACLE = (255, 0, 100)
    COLOR_OBSTACLE_GLOW = (128, 0, 50)
    COLOR_BEAT = (255, 255, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE_JUMP = (0, 255, 255)
    COLOR_PARTICLE_HIT = (255, 50, 100)
    
    GROUND_Y = 350
    PLAYER_X = 100
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT = 20
    GRAVITY = 0.8
    JUMP_STRENGTH = -14

    BEAT_INTERVAL = 15  # 0.5 seconds at 30 FPS
    BEAT_WINDOW = 3  # Frames to be considered "on beat"

    INITIAL_OBSTACLE_SPEED = 4.0
    OBSTACLE_SPEED_INCREASE = 0.2 # Increased from 0.05 for more noticeable difficulty scaling

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.obstacles = []
        self.particles = []
        self.player_pos = [0,0]
        self.player_y_vel = 0
        self.is_grounded = True
        self.score = 0
        self.steps = 0
        self.hits = 0
        self.game_over = False
        self.beat_timer = 0
        self.obstacle_speed = 0
        self.last_space_held = False
        
        self.reset()
        
        # This is a critical self-check
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.hits = 0
        self.game_over = False
        self.beat_timer = 0
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        self.last_space_held = False

        self.player_pos = [self.PLAYER_X, self.GROUND_Y - self.PLAYER_HEIGHT]
        self.player_y_vel = 0
        self.is_grounded = True

        self.particles.clear()
        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1  # Survival reward

        # --- Handle Input ---
        space_held = action[1] == 1
        jump_attempt = space_held and not self.last_space_held
        self.last_space_held = space_held

        # --- Update Game Logic ---
        self.steps += 1
        self.beat_timer = (self.beat_timer + 1) % self.BEAT_INTERVAL

        # --- Player Physics ---
        self.player_y_vel += self.GRAVITY
        self.player_pos[1] += self.player_y_vel
        
        if self.player_pos[1] >= self.GROUND_Y - self.PLAYER_HEIGHT:
            if not self.is_grounded:
                self._create_particles(self.player_pos[0] + self.PLAYER_WIDTH / 2, self.GROUND_Y, 5, self.COLOR_PLAYER)
                # Sound: Land
            self.player_pos[1] = self.GROUND_Y - self.PLAYER_HEIGHT
            self.player_y_vel = 0
            self.is_grounded = True
        else:
            self.is_grounded = False

        # --- Jump Logic ---
        is_on_beat = self.beat_timer < self.BEAT_WINDOW
        if jump_attempt and self.is_grounded:
            if is_on_beat:
                self.player_y_vel = self.JUMP_STRENGTH
                self.is_grounded = False
                reward += 1.0
                # Sound: Perfect Jump
                self._create_particles(self.player_pos[0] + self.PLAYER_WIDTH / 2, self.GROUND_Y, 15, self.COLOR_PARTICLE_JUMP)
            else:
                # Sound: Failed Jump
                self._create_particles(self.player_pos[0] + self.PLAYER_WIDTH / 2, self.GROUND_Y, 5, (100,100,100))


        # --- Obstacle Logic ---
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_speed += self.OBSTACLE_SPEED_INCREASE

        for obs in self.obstacles:
            obs.x -= self.obstacle_speed
        
        self.obstacles = [obs for obs in self.obstacles if obs.right > 0]

        # --- Collision Detection ---
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        collided_obstacles = []
        for obs in self.obstacles:
            if player_rect.colliderect(obs):
                self.hits += 1
                reward -= 5.0
                collided_obstacles.append(obs)
                # Sound: Hit
                self._create_particles(player_rect.centerx, player_rect.centery, 20, self.COLOR_PARTICLE_HIT, is_explosion=True)
        
        if collided_obstacles:
            self.obstacles = [obs for obs in self.obstacles if obs not in collided_obstacles]

        # --- Particle Update ---
        self._update_particles()

        # --- Termination Check ---
        terminated = False
        if self.hits >= 3:
            terminated = True
            reward = -100.0
            # Sound: Game Over
        elif self.steps >= self.LEVEL_END_STEP:
            terminated = True
            reward += 50.0
            # Sound: Level Complete
        elif self.steps >= self.MAX_STEPS:
            terminated = True # Time limit reached

        self.game_over = terminated
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "hits": self.hits,
            "progress": min(1.0, self.steps / self.LEVEL_END_STEP)
        }

    def _generate_level(self):
        self.obstacles.clear()
        current_x = self.WIDTH + 100
        level_pixel_length = self.obstacle_speed * self.LEVEL_END_STEP

        while current_x < level_pixel_length:
            gap = self.np_random.integers(150, 250)
            current_x += gap
            
            obstacle_height = self.np_random.integers(30, 80)
            obstacle_width = self.np_random.integers(20, 40)
            
            obstacle = pygame.Rect(
                current_x,
                self.GROUND_Y - obstacle_height,
                obstacle_width,
                obstacle_height
            )
            self.obstacles.append(obstacle)

    def _render_background(self):
        # Efficient gradient drawing
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # --- Ground ---
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 3)

        # --- Beat Indicator ---
        pulse_progress = self.beat_timer / self.BEAT_INTERVAL
        
        # Fast expansion, slow contraction
        if pulse_progress < (self.BEAT_WINDOW / self.BEAT_INTERVAL):
             # On beat window
            radius = 15 + 15 * (pulse_progress / (self.BEAT_WINDOW / self.BEAT_INTERVAL))
            alpha = 255
        else:
            # Off beat
            radius = 30 - 25 * ((pulse_progress - (self.BEAT_WINDOW/self.BEAT_INTERVAL)) / (1.0 - (self.BEAT_WINDOW/self.BEAT_INTERVAL)))
            alpha = 200 - 150 * pulse_progress
        
        pygame.gfxdraw.filled_circle(self.screen, 50, 50, int(radius), (*self.COLOR_BEAT, int(alpha)))
        pygame.gfxdraw.aacircle(self.screen, 50, 50, int(radius), (*self.COLOR_BEAT, int(alpha)))


        # --- Obstacles ---
        for obs in self.obstacles:
            # Glow effect
            glow_rect = obs.inflate(8, 8)
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*self.COLOR_OBSTACLE_GLOW, 50), glow_surface.get_rect(), border_radius=5)
            self.screen.blit(glow_surface, glow_rect.topleft)
            
            # Main shape
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs, border_radius=3)

        # --- Particles ---
        for p in self.particles:
            p_color = (*p['color'], int(255 * (p['life'] / p['max_life'])))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p_color)
            
        # --- Player ---
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        
        # Squash and stretch
        if self.is_grounded:
            squash = min(5, self.player_y_vel) # Not really used as vel is 0, but can be adapted
            player_rect.height = self.PLAYER_HEIGHT - squash
            player_rect.width = self.PLAYER_WIDTH + squash
            player_rect.y = self.GROUND_Y - player_rect.height
            player_rect.x = self.PLAYER_X - squash / 2
        else:
            stretch = min(10, max(-5, int(self.player_y_vel)))
            player_rect.height = self.PLAYER_HEIGHT + stretch
            player_rect.y = self.player_pos[1] - stretch / 2

        # Glow effect
        glow_rect = player_rect.inflate(10, 10)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, (*self.COLOR_PLAYER_GLOW, 80), glow_surface.get_rect(), border_radius=8)
        self.screen.blit(glow_surface, glow_rect.topleft)

        # Main shape
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)


    def _render_ui(self):
        # --- Score ---
        score_text = self.font_large.render(f"{int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 20))

        # --- Hits ---
        for i in range(3):
            color = self.COLOR_OBSTACLE if i < self.hits else (50, 50, 80)
            pygame.draw.rect(self.screen, color, (20 + i * 25, self.HEIGHT - 40, 20, 20), border_radius=3)

        # --- Progress Bar ---
        progress = self.steps / self.LEVEL_END_STEP
        bar_width = self.WIDTH - 40
        bar_height = 5
        
        pygame.draw.rect(self.screen, (50, 50, 80), (20, self.HEIGHT - 20, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (20, self.HEIGHT - 20, bar_width * progress, bar_height), border_radius=3)

    def _create_particles(self, x, y, count, color, is_explosion=False):
        for _ in range(count):
            if is_explosion:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 6)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            else: # Ground particles
                vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(-3, -1)]
            
            life = self.np_random.integers(10, 20)
            self.particles.append({
                'pos': [x, y],
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'life': life,
                'max_life': life,
                'color': color,
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # particle gravity
            p['radius'] -= 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]
        
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to run in a window
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'cocoa'

    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Rhythm Runner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action Mapping for Human Play ---
        # 0=none, 1=up, 2=down, 3=left, 4=right
        # space, shift
        action = [0, 0, 0] 
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- RESET ---")
                if event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Auto-reset on termination
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering to the display window ---
        # The observation is (H, W, C), but pygame wants (W, H) surface
        # We need to transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
    env.close()