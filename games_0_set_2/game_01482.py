
# Generated: 2025-08-27T17:17:11.027538
# Source Brief: brief_01482.md
# Brief Index: 1482

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ↑ and ↓ to steer your vehicle vertically. Dodge red obstacles and collect green speed boosts."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, retro-futuristic racer. Survive as long as possible by dodging obstacles and collecting boosts across three increasingly difficult stages."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (10, 0, 20)
    COLOR_TRACK = (70, 30, 100)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 200, 255, 50)
    COLOR_OBSTACLE_1 = (255, 50, 50)
    COLOR_OBSTACLE_2 = (255, 120, 50)
    COLOR_BOOST = (50, 255, 50)
    COLOR_BOOST_GLOW = (50, 255, 50, 60)
    COLOR_TEXT = (220, 220, 255)
    COLOR_TIMER_WARN = (255, 200, 0)
    COLOR_TIMER_CRIT = (255, 50, 50)

    # Player settings
    PLAYER_WIDTH, PLAYER_HEIGHT = 20, 15
    PLAYER_ACCEL = 1.5
    PLAYER_DRAG = 0.85
    PLAYER_MAX_VY = 10
    
    # Track settings
    TRACK_Y_TOP = 80
    TRACK_Y_BOTTOM = HEIGHT - 80
    TRACK_HEIGHT = TRACK_Y_BOTTOM - TRACK_Y_TOP
    
    # Game settings
    STAGE_LENGTH = 3000  # Arbitrary units of progress
    TIME_PER_STAGE = 60  # Seconds
    BOOST_DURATION = 2 * FPS
    BOOST_MULTIPLIER = 1.5
    
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
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Internal state - these are initialized in reset()
        self.np_random = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.player_y = 0
        self.player_vy = 0
        self.player_rect = None
        
        self.current_stage = 1
        self.stage_progress = 0
        self.base_speed = 0
        self.speed_multiplier = 1.0
        
        self.stage_timer = 0
        self.boost_timer = 0
        
        self.obstacles = None
        self.boosts = None
        self.particles = None
        self.stars = None
        
        self.last_obstacle_y = self.HEIGHT / 2
        self.obstacle_spawn_cooldown = 0
        self.boost_spawn_cooldown = 0

        # Initialize state variables
        self.reset()
        
        # Validate implementation after full initialization
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.player_y = self.HEIGHT / 2
        self.player_vy = 0
        self.player_rect = pygame.Rect(self.WIDTH * 0.2, self.player_y - self.PLAYER_HEIGHT / 2, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        
        self.current_stage = 1
        self.stage_progress = 0
        self.base_speed = 4
        self.speed_multiplier = 1.0
        
        self.stage_timer = self.TIME_PER_STAGE * self.FPS
        self.boost_timer = 0
        
        self.obstacles = deque()
        self.boosts = deque()
        self.particles = deque()
        
        self.last_obstacle_y = self.HEIGHT / 2
        self.obstacle_spawn_cooldown = 0
        self.boost_spawn_cooldown = self.FPS * 2 # Delay first boost

        if self.stars is None: # Only generate stars once
            self.stars = [
                {
                    "pos": [self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)],
                    "speed": self.np_random.uniform(0.1, 0.5),
                    "size": self.np_random.integers(1, 3),
                    "color": random.choice([(100,100,100), (150,150,150), (200,200,200)])
                }
                for _ in range(150)
            ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        terminated = False
        
        # 1. Unpack action
        movement = action[0]

        # 2. Calculate pre-update reward
        # Survival reward
        reward += 0.01 
        
        # "Safe action" penalty from brief (interpreted for better gameplay)
        # This penalizes moving away from a far-off obstacle, encouraging riskier play.
        closest_obs = self._get_closest_obstacle()
        if closest_obs and closest_obs['rect'].x > self.player_rect.right + 150:
            dy_before = abs(self.player_y - closest_obs['rect'].y)
            
            potential_vy = self.player_vy
            if movement == 1: # Up
                potential_vy -= self.PLAYER_ACCEL
            elif movement == 2: # Down
                potential_vy += self.PLAYER_ACCEL
            potential_vy = np.clip(potential_vy * self.PLAYER_DRAG, -self.PLAYER_MAX_VY, self.PLAYER_MAX_VY)
            potential_y = self.player_y + potential_vy
            
            dy_after = abs(potential_y - closest_obs['rect'].y)
            if dy_after > dy_before + 1: # If moved away
                reward -= 0.02

        # 3. Update game logic
        self._update_player(movement)
        self._update_world_state()
        self._update_entities()
        self._spawn_entities()
        
        # 4. Handle collisions and events
        collision_reward, terminated_by_collision = self._handle_collisions()
        reward += collision_reward
        if terminated_by_collision:
            terminated = True
        
        # 5. Check for other termination/progression conditions
        if not terminated:
            # Stage completion
            if self.stage_progress >= self.STAGE_LENGTH:
                self.current_stage += 1
                self.stage_progress = 0
                self.stage_timer = self.TIME_PER_STAGE * self.FPS
                
                if self.current_stage > 3:
                    reward += 300  # Game complete bonus
                    terminated = True
                else:
                    reward += 100  # Stage complete bonus
                    # Increase difficulty
                    self.base_speed += 1.5
            
            # Timeout
            if self.stage_timer <= 0:
                reward -= 100
                terminated = True
        
        self.game_over = terminated
        self.score += reward
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        if movement == 1:  # Up
            self.player_vy -= self.PLAYER_ACCEL
        elif movement == 2:  # Down
            self.player_vy += self.PLAYER_ACCEL
        
        self.player_vy *= self.PLAYER_DRAG
        self.player_vy = np.clip(self.player_vy, -self.PLAYER_MAX_VY, self.PLAYER_MAX_VY)
        
        self.player_y += self.player_vy
        self.player_y = np.clip(self.player_y, self.TRACK_Y_TOP + self.PLAYER_HEIGHT, self.TRACK_Y_BOTTOM - self.PLAYER_HEIGHT)
        
        self.player_rect.centery = int(self.player_y)

    def _update_world_state(self):
        current_speed = self.base_speed * self.speed_multiplier
        self.stage_progress += current_speed
        self.stage_timer -= 1
        
        if self.boost_timer > 0:
            self.boost_timer -= 1
            if self.boost_timer == 0:
                self.speed_multiplier = 1.0

    def _update_entities(self):
        current_speed = self.base_speed * self.speed_multiplier
        
        # Move obstacles
        for obs in self.obstacles:
            obs['rect'].x -= current_speed + obs['speed_offset']
        # Move boosts
        for boost in self.boosts:
            boost['rect'].x -= current_speed
        # Move particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            
        # Prune off-screen entities
        while self.obstacles and self.obstacles[0]['rect'].right < 0:
            self.obstacles.popleft()
        while self.boosts and self.boosts[0]['rect'].right < 0:
            self.boosts.popleft()
        while self.particles and self.particles[0]['life'] <= 0:
            self.particles.popleft()
            
    def _spawn_entities(self):
        # Spawn obstacles
        self.obstacle_spawn_cooldown -= 1
        if self.obstacle_spawn_cooldown <= 0:
            # Difficulty scaling from brief
            spawn_rates = {1: 20, 2: 15, 3: 10} # frames between spawns
            self.obstacle_spawn_cooldown = spawn_rates.get(self.current_stage, 10)
            
            size = self.np_random.integers(15, 35)
            is_circle = self.np_random.choice([True, False])
            
            # Ensure path exists by spawning away from last obstacle
            min_gap = 80
            y_range = list(range(self.TRACK_Y_TOP + size, int(self.last_obstacle_y - min_gap))) + \
                      list(range(int(self.last_obstacle_y + min_gap), self.TRACK_Y_BOTTOM - size))
            if not y_range: # Fallback if gap logic fails
                y_range = range(self.TRACK_Y_TOP + size, self.TRACK_Y_BOTTOM - size)
            
            y = self.np_random.choice(y_range)
            self.last_obstacle_y = y
            
            speed_offset = self.np_random.uniform(-1, self.current_stage)
            
            self.obstacles.append({
                'rect': pygame.Rect(self.WIDTH, y - size // 2, size, size),
                'color': random.choice([self.COLOR_OBSTACLE_1, self.COLOR_OBSTACLE_2]),
                'is_circle': is_circle,
                'speed_offset': speed_offset
            })

        # Spawn boosts
        self.boost_spawn_cooldown -= 1
        if self.boost_spawn_cooldown <= 0:
            self.boost_spawn_cooldown = self.np_random.integers(self.FPS * 3, self.FPS * 6)
            size = 20
            y = self.np_random.integers(self.TRACK_Y_TOP + 30, self.TRACK_Y_BOTTOM - 30)
            self.boosts.append({
                'rect': pygame.Rect(self.WIDTH, y - size // 2, size, size)
            })
            
    def _handle_collisions(self):
        reward = 0
        terminated = False
        
        # Player-Obstacle collision
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs['rect']):
                # sfx: explosion
                reward -= 100
                terminated = True
                self._create_explosion(self.player_rect.center, self.COLOR_PLAYER)
                break
        
        # Player-Boost collision
        if not terminated:
            collected_boosts = []
            for i, boost in enumerate(self.boosts):
                if self.player_rect.colliderect(boost['rect']):
                    # sfx: boost collect
                    reward += 10
                    self.speed_multiplier = self.BOOST_MULTIPLIER
                    self.boost_timer = self.BOOST_DURATION
                    collected_boosts.append(i)
                    self._create_explosion(boost['rect'].center, self.COLOR_BOOST, 20)
            
            for i in sorted(collected_boosts, reverse=True):
                self.boosts.remove(self.boosts[i])

        return reward, terminated
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_track()
        self._render_entities()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        progress_factor = self.stage_progress / self.STAGE_LENGTH
        for star in self.stars:
            x = (star['pos'][0] - (self.stage_progress * star['speed'])) % self.WIDTH
            y = star['pos'][1]
            pygame.draw.circle(self.screen, star['color'], (int(x), int(y)), star['size'])

    def _render_track(self):
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_TOP), (self.WIDTH, self.TRACK_Y_TOP), 5)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_BOTTOM), (self.WIDTH, self.TRACK_Y_BOTTOM), 5)

        # Progress bar
        progress_width = (self.stage_progress / self.STAGE_LENGTH) * self.WIDTH
        if progress_width > 0:
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (0, self.TRACK_Y_TOP), (int(progress_width), self.TRACK_Y_TOP), 5)
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (0, self.TRACK_Y_BOTTOM), (int(progress_width), self.TRACK_Y_BOTTOM), 5)


    def _render_entities(self):
        # Render boosts
        for boost in self.boosts:
            center = boost['rect'].center
            radius = boost['rect'].width // 2
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius + 4, self.COLOR_BOOST_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, self.COLOR_BOOST)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, self.COLOR_BOOST)

        # Render obstacles
        for obs in self.obstacles:
            if obs['is_circle']:
                pygame.gfxdraw.filled_circle(self.screen, obs['rect'].centerx, obs['rect'].centery, obs['rect'].width // 2, obs['color'])
                pygame.gfxdraw.aacircle(self.screen, obs['rect'].centerx, obs['rect'].centery, obs['rect'].width // 2, obs['color'])
            else:
                pygame.draw.rect(self.screen, obs['color'], obs['rect'])
    
    def _render_player(self):
        if self.game_over:
            return

        p = self.player_rect
        points = [
            (p.right, p.centery),
            (p.left, p.top),
            (p.left, p.bottom)
        ]

        # Glow effect
        glow_points = [
            (p.right + 5, p.centery),
            (p.left - 5, p.top - 5),
            (p.left - 5, p.bottom + 5)
        ]
        pygame.gfxdraw.filled_trigon(self.screen, glow_points[0][0], glow_points[0][1], glow_points[1][0], glow_points[1][1], glow_points[2][0], glow_points[2][1], self.COLOR_PLAYER_GLOW)
        
        # Main ship
        pygame.gfxdraw.filled_trigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], self.COLOR_PLAYER)
        pygame.gfxdraw.aatrigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], self.COLOR_PLAYER)

    def _render_particles(self):
        # Player trail
        if not self.game_over and self.np_random.random() < 0.7:
            life = self.np_random.integers(10, 20)
            pos = [self.player_rect.left, self.player_rect.centery + self.np_random.uniform(-5, 5)]
            vel = [-self.np_random.uniform(1, 3), self.np_random.uniform(-0.5, 0.5)]
            color = self.COLOR_PLAYER if self.boost_timer == 0 else self.COLOR_BOOST
            self.particles.append({'pos': pos, 'vel': vel, 'life': life, 'max_life': life, 'color': color})

        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            size = int(5 * (p['life'] / p['max_life']))
            if size > 0:
                surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (size, size), size)
                self.screen.blit(surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

    def _render_ui(self):
        # Stage display
        stage_text = self.font_medium.render(f"STAGE {self.current_stage}/3", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (20, 10))

        # Timer display
        time_left = max(0, self.stage_timer / self.FPS)
        time_color = self.COLOR_TEXT
        if time_left < 10: time_color = self.COLOR_TIMER_CRIT
        elif time_left < 20: time_color = self.COLOR_TIMER_WARN
        time_text = self.font_medium.render(f"TIME: {time_left:.1f}", True, time_color)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 20, 10))
        
        # Score display
        score_text = self.font_small.render(f"SCORE: {self.score:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 40))

        # Speed multiplier display
        if self.boost_timer > 0:
            boost_text = self.font_large.render(f"x{self.speed_multiplier:.1f} BOOST!", True, self.COLOR_BOOST)
            text_rect = boost_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 40))
            self.screen.blit(boost_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "time_left": max(0, self.stage_timer / self.FPS),
        }
        
    def _create_explosion(self, position, color, num_particles=50):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(20, 40)
            self.particles.append({'pos': list(position), 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _get_closest_obstacle(self):
        closest = None
        min_dist = float('inf')
        for obs in self.obstacles:
            if obs['rect'].right > self.player_rect.left:
                dist = obs['rect'].left - self.player_rect.right
                if dist < min_dist:
                    min_dist = dist
                    closest = obs
        return closest

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        print("🔬 Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
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

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "human" to see the game being played
    render_mode = "human" # "rgb_array" or "human"
    
    if render_mode == "human":
        # For human playback, we need a real display
        import os
        if os.environ.get("SDL_VIDEODRIVER", "") == "dummy":
            del os.environ["SDL_VIDEODRIVER"]
        
        env = GameEnv(render_mode="rgb_array")
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption(GameEnv.game_description)
        clock = pygame.time.Clock()
    else:
        env = GameEnv(render_mode="rgb_array")

    obs, info = env.reset()
    done = False
    
    print(env.user_guide)

    total_reward = 0
    while not done:
        if render_mode == "human":
            # --- Human Controls ---
            action = [0, 0, 0] # no-op, released, released
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            
            # The brief requires these actions, even if unused in this game
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        else:
            # --- Agent Controls ---
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if render_mode == "human":
            # Transpose obs from (H, W, C) to (W, H, C) for pygame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(env.FPS)

    print(f"Game Over. Final Score: {total_reward:.2f}, Steps: {info['steps']}")
    env.close()