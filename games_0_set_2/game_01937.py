
# Generated: 2025-08-28T03:09:43.022998
# Source Brief: brief_01937.md
# Brief Index: 1937

        
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
        "Controls: Use ← and → to move, and ↑ to jump. Avoid the incoming shapes."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-view arcade game where you control a hopping spaceship. "
        "Dodge obstacles to reach the end of the level for a time bonus."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 50
    GROUND_Y = 350
    LEVEL_END_X = 600

    # Colors
    COLOR_BG = (10, 15, 30)
    COLOR_PLAYER = (60, 160, 255)
    COLOR_PLAYER_GLOW = (120, 200, 255)
    COLOR_GROUND = (40, 50, 70)
    COLOR_OBSTACLE_TYPES = [(255, 80, 80), (80, 255, 80), (255, 255, 80)]
    COLOR_END_MARKER = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)

    # Physics
    GRAVITY = 0.5
    JUMP_STRENGTH = -11
    PLAYER_SPEED = 4
    PLAYER_SIZE = (20, 20)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Initialize state variables
        self.player_pos = [0, 0]
        self.player_vel_y = 0
        self.on_ground = True
        self.obstacles = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.time_remaining = 0
        self.obstacle_spawn_timer = 0
        self.obstacle_id_counter = 0
        self.jumped_obstacles = set()
        self.obstacle_speed_modifier = 1.0
        self.screen_shake = 0

        # This will be properly initialized in reset()
        self.np_random = None

        self.validate_implementation()

    def _create_stars(self):
        self.stars = []
        for i in range(150):
            layer = self.np_random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            self.stars.append({
                'pos': [self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)],
                'size': self.np_random.integers(1, 3),
                'layer': layer, # 1=slow, 2=medium, 3=fast
                'brightness': self.np_random.integers(50, 150)
            })

    def _start_next_level(self):
        self.player_pos = [100, self.GROUND_Y - self.PLAYER_SIZE[1]]
        self.player_vel_y = 0
        self.on_ground = True
        self.obstacles.clear()
        self.particles.clear()
        self.jumped_obstacles.clear()
        self.time_remaining = 60 * self.FPS
        self.obstacle_spawn_timer = self.np_random.integers(40, 80)
        self.obstacle_id_counter = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level = 1
        self.obstacle_speed_modifier = 1.0
        self.screen_shake = 0
        self._create_stars()
        self._start_next_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, _, _ = action
        reward = 0.1  # Survival reward

        # --- Update Player ---
        if movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        elif movement == 0: # No horizontal movement
            reward -= 0.2
        
        if movement == 1 and self.on_ground:  # Jump
            self.player_vel_y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: player_jump.wav
            for _ in range(15):
                self._spawn_particle(
                    pos=[self.player_pos[0] + self.PLAYER_SIZE[0] / 2, self.player_pos[1] + self.PLAYER_SIZE[1]],
                    vel=[(self.np_random.random() - 0.5) * 3, self.np_random.random() * 3 + 1],
                    life=self.np_random.integers(10, 20),
                    color=self.COLOR_PLAYER_GLOW,
                    radius=self.np_random.integers(2, 4)
                )

        self.player_vel_y += self.GRAVITY
        self.player_pos[1] += self.player_vel_y
        
        # Clamp player position
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH - self.PLAYER_SIZE[0])
        
        if self.player_pos[1] >= self.GROUND_Y - self.PLAYER_SIZE[1]:
            self.player_pos[1] = self.GROUND_Y - self.PLAYER_SIZE[1]
            if not self.on_ground: # Landing
                self.screen_shake = 5
            self.on_ground = True
            self.player_vel_y = 0

        # --- Update Game State ---
        self.time_remaining -= 1
        self.steps += 1
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_speed_modifier += 0.2
        
        self._update_obstacles(reward)
        self._update_particles()
        self._update_stars()

        # --- Termination and Rewards ---
        terminated = False
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.PLAYER_SIZE[0], self.PLAYER_SIZE[1])
        
        for obs in self.obstacles:
            if player_rect.colliderect(obs['rect']):
                self.game_over = True
                terminated = True
                reward -= 50
                self.screen_shake = 20
                # sfx: explosion.wav
                for _ in range(50):
                    angle = self.np_random.random() * 2 * math.pi
                    speed = self.np_random.random() * 8
                    self._spawn_particle(
                        pos=list(player_rect.center),
                        vel=[math.cos(angle) * speed, math.sin(angle) * speed],
                        life=self.np_random.integers(20, 40),
                        color=self.np_random.choice(self.COLOR_OBSTACLE_TYPES),
                        radius=self.np_random.integers(2, 6)
                    )
                break
        
        # Check for successful jump reward
        for obs in self.obstacles:
            if obs['id'] not in self.jumped_obstacles and player_rect.left > obs['rect'].right:
                reward += 1.0
                self.jumped_obstacles.add(obs['id'])
                # sfx: reward_ping.wav

        if self.player_pos[0] >= self.LEVEL_END_X:
            reward += 100
            self.level += 1
            if self.level > 3:
                self.game_over = True
                terminated = True
            else:
                self._start_next_level()
        
        if self.time_remaining <= 0 and not terminated:
            self.game_over = True
            terminated = True
            reward -= 10
        
        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_particle(self, pos, vel, life, color, radius):
        self.particles.append({'pos': list(pos), 'vel': list(vel), 'life': life, 'max_life': life, 'color': color, 'radius': radius})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _spawn_obstacle(self):
        obstacle_type = self.np_random.integers(0, min(self.level, 3))
        y_pos = self.GROUND_Y - 30
        size = [30, 30]
        speed = self.np_random.uniform(2, 4) * self.obstacle_speed_modifier
        
        if obstacle_type == 0: # Rectangle
            y_pos = self.GROUND_Y - 40
            size = [self.np_random.integers(20, 40), 40]
        elif obstacle_type == 1: # Triangle (flying)
            y_pos = self.np_random.integers(self.GROUND_Y - 120, self.GROUND_Y - 50)
            size = [30, 30]
        elif obstacle_type == 2: # Circle (bouncing)
            y_pos = self.GROUND_Y - 30
            size = [30, 30]

        self.obstacles.append({
            'id': self.obstacle_id_counter,
            'type': obstacle_type,
            'pos': [self.WIDTH, y_pos],
            'size': size,
            'color': self.COLOR_OBSTACLE_TYPES[obstacle_type],
            'speed': speed,
            'vel_y': -4 if obstacle_type == 2 else 0,
            'phase': self.np_random.random() * 2 * math.pi if obstacle_type == 1 else 0,
            'rect': pygame.Rect(self.WIDTH, y_pos, size[0], size[1])
        })
        self.obstacle_id_counter += 1
        self.obstacle_spawn_timer = self.np_random.integers(max(20, 60 - self.level * 10), max(40, 100 - self.level * 15))

    def _update_obstacles(self, reward):
        self.obstacle_spawn_timer -= 1
        if self.obstacle_spawn_timer <= 0:
            self._spawn_obstacle()

        for obs in self.obstacles:
            obs['pos'][0] -= obs['speed']
            if obs['type'] == 1: # Sinusoidal
                obs['pos'][1] += math.sin(self.steps * 0.1 + obs['phase']) * 2
            elif obs['type'] == 2: # Bouncing
                obs['vel_y'] += self.GRAVITY / 2
                obs['pos'][1] += obs['vel_y']
                if obs['pos'][1] >= self.GROUND_Y - obs['size'][1]:
                    obs['pos'][1] = self.GROUND_Y - obs['size'][1]
                    obs['vel_y'] = -abs(obs['vel_y']) * 0.8
            obs['rect'].topleft = obs['pos']
        
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]

    def _update_stars(self):
        for star in self.stars:
            star['pos'][0] -= 0.1 * star['layer'] # Parallax effect
            if star['pos'][0] < 0:
                star['pos'][0] = self.WIDTH
                star['pos'][1] = self.np_random.integers(0, self.HEIGHT)

    def _get_observation(self):
        # Apply screen shake
        render_offset = [0, 0]
        if self.screen_shake > 0:
            self.screen_shake -= 1
            render_offset[0] = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            render_offset[1] = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
        
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background(render_offset)
        self._render_game(render_offset)
        
        # Render UI overlay (not affected by shake)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, offset):
        for star in self.stars:
            color_val = star['brightness']
            if self.np_random.random() < 0.01: # Twinkle
                color_val = 255
            color = (color_val, color_val, color_val)
            pygame.draw.circle(self.screen, color, (int(star['pos'][0] + offset[0]), int(star['pos'][1] + offset[1])), star['size'])
        
        # Ground
        ground_rect = pygame.Rect(0 + offset[0], self.GROUND_Y + offset[1], self.WIDTH, self.HEIGHT - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)

    def _render_game(self, offset):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            pos = (int(p['pos'][0] + offset[0]), int(p['pos'][1] + offset[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p['radius']), color)

        # End Marker
        if self.level <= 3:
            end_x = int(self.LEVEL_END_X + offset[0])
            pygame.draw.line(self.screen, self.COLOR_END_MARKER, (end_x, 0), (end_x, self.GROUND_Y), 2)
            for i in range(10):
                y = int(self.GROUND_Y - i*20 - (self.steps % 20))
                pygame.draw.line(self.screen, self.COLOR_END_MARKER, (end_x, y), (end_x+10, y-10), 2)

        # Obstacles
        for obs in self.obstacles:
            r = obs['rect'].move(offset)
            if obs['type'] == 0: # Rectangle
                pygame.gfxdraw.box(self.screen, r, obs['color'])
            elif obs['type'] == 1: # Triangle
                points = [(r.left, r.bottom), (r.centerx, r.top), (r.right, r.bottom)]
                pygame.gfxdraw.aapolygon(self.screen, points, obs['color'])
                pygame.gfxdraw.filled_polygon(self.screen, points, obs['color'])
            elif obs['type'] == 2: # Circle
                pygame.gfxdraw.filled_circle(self.screen, r.centerx, r.centery, r.width // 2, obs['color'])
                pygame.gfxdraw.aacircle(self.screen, r.centerx, r.centery, r.width // 2, obs['color'])

        # Player
        if not self.game_over:
            px, py = self.player_pos
            w, h = self.PLAYER_SIZE
            
            # Hopping animation
            bob = math.sin(self.steps * 0.2) * 2 if self.on_ground else 0
            
            player_center = (int(px + w/2 + offset[0]), int(py + h/2 + offset[1] + bob))
            
            # Glow effect
            glow_radius = int(w * 0.8 + abs(math.sin(self.steps * 0.1) * 3))
            glow_color = self.COLOR_PLAYER_GLOW + (50,)
            pygame.gfxdraw.filled_circle(self.screen, player_center[0], player_center[1], glow_radius, glow_color)

            # Spaceship body
            points = [
                (px + w/2, py + bob), 
                (px + w, py + h*0.7 + bob),
                (px, py + h*0.7 + bob)
            ]
            points = [(p[0] + offset[0], p[1] + offset[1]) for p in points]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Time
        time_text = f"TIME: {max(0, self.time_remaining // self.FPS):02d}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Level
        level_text = f"LEVEL: {self.level}"
        level_surf = self.font_small.render(level_text, True, self.COLOR_TEXT)
        self.screen.blit(level_surf, (self.WIDTH - level_surf.get_width() - 10, 10))
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH // 2 - score_surf.get_width() // 2, 10))

        if self.game_over:
            msg = "GAME OVER" if self.level <= 3 else "YOU WIN!"
            end_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(end_surf, (self.WIDTH // 2 - end_surf.get_width() // 2, self.HEIGHT // 2 - end_surf.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "time_remaining": self.time_remaining,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Reset first to ensure everything is initialized
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Override the screen to be a display window
    env.screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Hopping Spaceship")

    terminated = False
    total_reward = 0
    
    # --- Main Game Loop ---
    while not terminated:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = [movement, 0, 0] # space and shift are unused

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # _get_observation already rendered to env.screen, so we just flip
        pygame.display.flip()
        
        # Cap the frame rate
        env.clock.tick(env.FPS)

    print(f"Game Over! Final Score: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()