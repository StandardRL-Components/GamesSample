
# Generated: 2025-08-27T21:32:15.608360
# Source Brief: brief_02820.md
# Brief Index: 2820

        
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
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to run, ↑ to jump. Reach the green finish line as fast as you can!"
    )

    game_description = (
        "A fast-paced, side-scrolling platformer. Guide your robot through a neon-drenched, "
        "procedurally generated world. Jump over deadly red obstacles and grab yellow "
        "boosters to maximize your score before time runs out."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 50
    MAX_STEPS = 30 * FPS  # 30 seconds
    LEVEL_LENGTH = 12000

    # Colors
    COLOR_BG = (20, 10, 40)
    COLOR_GRID = (40, 20, 80)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_BOOST = (255, 255, 0)
    COLOR_PLAYER_HEAD = (255, 255, 255)
    COLOR_PLATFORM = (60, 60, 80)
    COLOR_PLATFORM_EDGE = (120, 120, 160)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (255, 100, 100, 100)
    COLOR_BONUS = (255, 255, 0)
    COLOR_BONUS_GLOW = (255, 255, 100, 100)
    COLOR_FINISH = (0, 255, 100)
    COLOR_FINISH_GLOW = (100, 255, 150, 100)

    # Physics
    GRAVITY = 0.8
    MOVE_SPEED = 6
    FRICTION = 0.85
    JUMP_STRENGTH = 15
    BOOST_MULTIPLIER = 1.5
    BOOST_DURATION = 3 * FPS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.game_over_message = ""
        self.player = None
        self.platforms = []
        self.obstacles = []
        self.bonuses = []
        self.particles = []
        self.camera_x = 0.0

        self.reset()
        
        # self.validate_implementation() # Uncomment for self-check

    def _generate_level(self):
        self.platforms = []
        self.obstacles = []
        self.bonuses = []

        current_x = 0
        current_y = self.HEIGHT - 80

        # Initial platform
        self.platforms.append(pygame.Rect(0, current_y, 800, self.HEIGHT - current_y))
        current_x = 800

        while current_x < self.LEVEL_LENGTH - 1000:
            gap_width = self.np_random.uniform(80, 200)
            current_x += gap_width

            platform_length = self.np_random.uniform(200, 600)
            height_change = self.np_random.uniform(-100, 100)
            current_y = np.clip(current_y + height_change, 150, self.HEIGHT - 50)
            
            new_platform = pygame.Rect(current_x, current_y, platform_length, self.HEIGHT - current_y)
            self.platforms.append(new_platform)

            if self.np_random.random() < 0.5:
                obs_x = current_x + self.np_random.uniform(0.2, 0.8) * platform_length
                obs_h = self.np_random.uniform(20, 40)
                self.obstacles.append(pygame.Rect(obs_x, current_y - obs_h, 20, obs_h))

            if self.np_random.random() < 0.3:
                bonus_x = current_x + self.np_random.uniform(0.2, 0.8) * platform_length
                bonus_y = current_y - self.np_random.uniform(80, 150)
                self.bonuses.append(pygame.Rect(bonus_x, bonus_y, 20, 20))

            current_x += platform_length

        self.finish_line_x = current_x + 500
        final_platform_y = current_y
        self.platforms.append(pygame.Rect(current_x, final_platform_y, 1500, self.HEIGHT - final_platform_y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player = self.Player(100, self.HEIGHT - 150, self)
        self._generate_level()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.time_left = self.MAX_STEPS
        
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        reward = 0.1  # Survival reward
        reward -= 0.02 # Small penalty to encourage efficiency

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        self.steps += 1
        self.time_left -= 1

        # --- Handle Input ---
        move_direction = 0
        if movement == 3:  # Left
            move_direction = -1
        elif movement == 4:  # Right
            move_direction = 1
        
        self.player.update(move_direction, movement == 1)

        # --- Update Game State ---
        self._update_particles()
        
        # Check for bonus collection
        collected_bonuses = self.player.rect.collidelistall(self.bonuses)
        if collected_bonuses:
            for i in sorted(collected_bonuses, reverse=True):
                # SFX: Bonus collect
                self._create_particles(self.bonuses[i].center, self.COLOR_BONUS, 20)
                del self.bonuses[i]
                self.score += 5
                reward += 5
                self.player.activate_boost()

        # --- Check Termination Conditions ---
        # 1. Reached finish line
        if self.player.rect.centerx >= self.finish_line_x:
            self.game_over = True
            self.game_over_message = "LEVEL COMPLETE!"
            self.score += 100
            reward = 100
        
        # 2. Fell off world
        elif self.player.rect.top > self.HEIGHT:
            self.game_over = True
            self.game_over_message = "FALLEN"
            reward = -50
        
        # 3. Hit obstacle
        elif self.player.rect.collidelist(self.obstacles) != -1:
            self.game_over = True
            self.game_over_message = "CRASHED"
            # SFX: Explosion
            self._create_particles(self.player.rect.center, self.COLOR_OBSTACLE, 50)
            reward = -50

        # 4. Ran out of time
        elif self.time_left <= 0:
            self.game_over = True
            self.game_over_message = "TIME'S UP"
            reward = -10

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_camera()
        self._render_background()
        self._render_game_elements()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "player_x": self.player.rect.x,
        }

    def _update_camera(self):
        self.camera_x = self.player.pos[0] - self.WIDTH / 3

    def _render_background(self):
        # Parallax grid
        for i in range(0, self.WIDTH, 50):
            x = i - (self.camera_x * 0.5) % 50
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 50):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

    def _render_game_elements(self):
        # Platforms
        for plat in self.platforms:
            screen_rect = plat.move(-self.camera_x, 0)
            if screen_rect.right < 0 or screen_rect.left > self.WIDTH:
                continue
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect)
            pygame.draw.line(self.screen, self.COLOR_PLATFORM_EDGE, 
                             (screen_rect.left, screen_rect.top), 
                             (screen_rect.right, screen_rect.top), 2)
        # Obstacles
        for obs in self.obstacles:
            screen_rect = obs.move(-self.camera_x, 0)
            if screen_rect.right < 0 or screen_rect.left > self.WIDTH:
                continue
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)
            pygame.gfxdraw.rectangle(self.screen, screen_rect, self.COLOR_OBSTACLE_GLOW)
        
        # Bonuses
        for bonus in self.bonuses:
            screen_rect = bonus.move(-self.camera_x, 0)
            if screen_rect.right < 0 or screen_rect.left > self.WIDTH:
                continue
            pygame.draw.rect(self.screen, self.COLOR_BONUS, screen_rect)
            pygame.gfxdraw.rectangle(self.screen, screen_rect, self.COLOR_BONUS_GLOW)

        # Finish line
        finish_screen_x = self.finish_line_x - self.camera_x
        if 0 < finish_screen_x < self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_screen_x, 0), (finish_screen_x, self.HEIGHT), 5)
            # Add a simple glow
            for i in range(1, 10, 2):
                alpha = 150 - i * 15
                pygame.draw.line(self.screen, (*self.COLOR_FINISH, alpha), (finish_screen_x - i, 0), (finish_screen_x - i, self.HEIGHT), 1)
                pygame.draw.line(self.screen, (*self.COLOR_FINISH, alpha), (finish_screen_x + i, 0), (finish_screen_x + i, self.HEIGHT), 1)


    def _render_player(self):
        if self.game_over and (self.game_over_message in ["CRASHED", "FALLEN"]):
            return # Don't draw player if they crashed or fell

        p_rect = self.player.rect.move(-self.camera_x, 0)
        
        # Trail effect
        for i, pos in enumerate(self.player.trail):
            alpha = int(100 * (i / len(self.player.trail)))
            trail_color = (*self.player.get_color(), alpha)
            trail_rect = pygame.Rect(pos[0] - self.camera_x, pos[1], self.player.rect.width, self.player.rect.height)
            s = pygame.Surface(trail_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, trail_color, s.get_rect())
            self.screen.blit(s, trail_rect.topleft)

        # Main body
        pygame.draw.rect(self.screen, self.player.get_color(), p_rect)
        
        # Headlight
        head_rect = pygame.Rect(0, 0, 8, 8)
        head_rect.center = p_rect.center
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_HEAD, head_rect)

    def _render_particles(self):
        for p in self.particles:
            p_pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            pygame.draw.circle(self.screen, p['color'], p_pos, int(p['size']))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Time
        time_str = f"TIME: {self.time_left // self.FPS:02d}"
        time_color = (255, 255, 255) if self.time_left > 5 * self.FPS else (255, 100, 100)
        time_text = self.font_small.render(time_str, True, time_color)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        
        # Boost indicator
        if self.player.boost_timer > 0:
            boost_text = self.font_small.render("BOOST!", True, self.COLOR_BONUS)
            self.screen.blit(boost_text, (self.WIDTH // 2 - boost_text.get_width() // 2, 10))

        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            over_text = self.font_large.render(self.game_over_message, True, (255, 255, 255))
            text_rect = over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)],
                'size': self.np_random.uniform(2, 6),
                'lifespan': self.np_random.integers(20, 40),
                'color': color
            })
    
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['size'] -= 0.1
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['size'] > 0]

    def close(self):
        pygame.quit()

    class Player:
        def __init__(self, x, y, env):
            self.env = env
            self.pos = [float(x), float(y)]
            self.vel = [0.0, 0.0]
            self.rect = pygame.Rect(x, y, 20, 30)
            self.on_ground = False
            self.boost_timer = 0
            self.trail = deque(maxlen=10)

        def update(self, move_direction, jump_pressed):
            # Horizontal movement
            current_speed = self.env.MOVE_SPEED * (self.env.BOOST_MULTIPLIER if self.boost_timer > 0 else 1)
            self.vel[0] += move_direction * current_speed * 0.2
            self.vel[0] *= self.env.FRICTION

            # Jumping
            if jump_pressed and self.on_ground:
                # SFX: Jump
                self.vel[1] = -self.env.JUMP_STRENGTH
                self.on_ground = False
                self.env._create_particles((self.rect.midbottom), self.env.COLOR_PLATFORM_EDGE, 5)


            # Apply gravity
            self.vel[1] += self.env.GRAVITY
            
            # Update position
            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]
            
            self.rect.x = int(self.pos[0])
            self.rect.y = int(self.pos[1])
            
            self.on_ground = False
            self._handle_collisions()

            # Update boost timer
            if self.boost_timer > 0:
                self.boost_timer -= 1
            
            # Update trail
            if self.env.steps % 2 == 0:
                self.trail.append(self.rect.topleft)

        def _handle_collisions(self):
            # Platform collisions
            for plat in self.env.platforms:
                if self.rect.colliderect(plat):
                    # Check if landing on top
                    if self.vel[1] > 0 and self.rect.bottom - self.vel[1] <= plat.top + 1:
                        if not self.on_ground: # First frame of landing
                            # SFX: Land
                            self.env._create_particles(self.rect.midbottom, self.env.COLOR_PLATFORM_EDGE, 10)
                        self.on_ground = True
                        self.vel[1] = 0
                        self.pos[1] = plat.top - self.rect.height
                        self.rect.bottom = plat.top
                    # Hitting side of platform
                    elif self.vel[0] > 0 and self.rect.right - self.vel[0] < plat.left:
                        self.vel[0] = 0
                        self.pos[0] = plat.left - self.rect.width
                    elif self.vel[0] < 0 and self.rect.left - self.vel[0] > plat.right:
                        self.vel[0] = 0
                        self.pos[0] = plat.right
            
            # World bounds
            if self.pos[0] < 0:
                self.pos[0] = 0
                self.vel[0] = 0

        def activate_boost(self):
            self.boost_timer = self.env.BOOST_DURATION

        def get_color(self):
            return self.env.COLOR_PLAYER_BOOST if self.boost_timer > 0 else self.env.COLOR_PLAYER

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("RoboRun")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping for human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame rendering ---
        # The observation is already a rendered image, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            print("Press 'R' to restart.")
            
        clock.tick(GameEnv.FPS)

    env.close()