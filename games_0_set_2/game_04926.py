import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Press ↑ or Space to jump. Hold to jump higher. The robot runs automatically."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Race a robot through an obstacle course. Collect coins for points, avoid red blocks, and reach the green finish line before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    # Colors
    COLOR_BG = (44, 62, 80) # Dark blue-grey
    COLOR_GROUND = (52, 73, 94) # Slightly lighter grey
    COLOR_ROBOT = (52, 152, 219) # Bright blue
    COLOR_ROBOT_EYE = (236, 240, 241) # White
    COLOR_OBSTACLE = (231, 76, 60) # Bright red
    COLOR_COIN = (241, 196, 15) # Bright yellow
    COLOR_COIN_SHINE = (243, 229, 171) # Lighter yellow
    COLOR_FINISH = (46, 204, 113) # Bright green
    COLOR_TEXT = (236, 240, 241) # White/light grey
    # Physics
    GRAVITY = 0.8
    JUMP_INITIAL_VELOCITY = -14
    JUMP_BOOST_POWER = -1.2
    MAX_JUMP_BOOST_FRAMES = 8
    # Game parameters
    MAX_STEPS = 900 # 30 seconds at 30 FPS
    MAX_LIVES = 3
    ROBOT_X_POS = 100
    GROUND_Y = 320
    WORLD_LENGTH = 15000
    INITIAL_SCROLL_SPEED = 6.0
    # Object generation
    CHUNK_SIZE = 800

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Initialize state variables
        # self.reset() is called by the wrapper/runner, no need to call it here.
        # However, we need to initialize some attributes for the validation to pass.
        self.np_random = None # Will be set by reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.MAX_LIVES
        self.world_offset_x = 0
        self.world_scroll_speed = self.INITIAL_SCROLL_SPEED
        self.robot_pos = pygame.Vector2(self.ROBOT_X_POS, self.GROUND_Y)
        self.robot_vel_y = 0
        self.robot_size = pygame.Vector2(30, 50)
        self.on_ground = True
        self.jump_boost_frames = 0
        self.invincibility_frames = 0
        self.squash_effect = 0
        self.last_action = np.array([0, 0, 0])
        self.obstacles = []
        self.coins = []
        self.particles = []
        self.generated_chunks = set()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lives = self.MAX_LIVES
        self.world_offset_x = 0
        self.world_scroll_speed = self.INITIAL_SCROLL_SPEED
        
        # Robot state
        self.robot_pos = pygame.Vector2(self.ROBOT_X_POS, self.GROUND_Y)
        self.robot_vel_y = 0
        self.robot_size = pygame.Vector2(30, 50)
        self.on_ground = True
        self.jump_boost_frames = 0
        self.invincibility_frames = 0
        self.squash_effect = 0

        # Action state
        self.last_action = np.array([0, 0, 0])

        # World objects
        self.obstacles = []
        self.coins = []
        self.particles = []
        self.generated_chunks = set()

        # Procedurally generate the initial world
        self._generate_world()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.1  # Base reward for surviving

        # 1. Handle Input
        self._handle_input(movement, space_held)

        # 2. Update Game Logic
        self._update_robot_physics()
        self._update_world_scroll()
        self._update_particles()
        
        # 3. Collision Detection and Interaction
        reward += self._handle_collisions()

        # 4. Object Lifecycle
        self._manage_world_generation()

        # 5. Update Timers and Counters
        self.steps += 1
        if self.invincibility_frames > 0:
            self.invincibility_frames -= 1
        if self.squash_effect > 0:
            self.squash_effect -= 0.1
            self.squash_effect = max(0, self.squash_effect)

        # 6. Check Termination Conditions
        terminated = self._check_termination()
        if terminated:
            if self.lives <= 0:
                reward = -100
            elif self.world_offset_x + self.robot_pos.x >= self.WORLD_LENGTH:
                reward += 50 # Victory bonus

        # Store last action for press detection
        self.last_action = action
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, move_action, space_held):
        # Detect press vs hold for jump
        jump_pressed = (space_held and self.last_action[1] == 0) or \
                       (move_action == 1 and self.last_action[0] != 1)

        # Initiate Jump
        if jump_pressed and self.on_ground:
            self.robot_vel_y = self.JUMP_INITIAL_VELOCITY
            self.on_ground = False
            self.jump_boost_frames = self.MAX_JUMP_BOOST_FRAMES
            self.squash_effect = 1.0 # Stretch for jump
            # sfx: jump_sound()
            self._create_particles(self.robot_pos + pygame.Vector2(0, self.robot_size.y/2), 10, self.COLOR_GROUND, (-2, 2), (-5, -1))

        # Apply Jump Boost (variable height)
        is_boosting = (space_held or move_action == 1)
        if is_boosting and self.robot_vel_y < 0 and self.jump_boost_frames > 0:
            self.robot_vel_y += self.JUMP_BOOST_POWER
            self.jump_boost_frames -= 1
            # sfx: boost_sound()
            self._create_particles(self.robot_pos + pygame.Vector2(0, self.robot_size.y/2), 1, self.COLOR_ROBOT, (-1, 1), (1, 3))

    def _update_robot_physics(self):
        if not self.on_ground:
            self.robot_vel_y += self.GRAVITY
        self.robot_pos.y += self.robot_vel_y

        if self.robot_pos.y >= self.GROUND_Y:
            self.robot_pos.y = self.GROUND_Y
            self.robot_vel_y = 0
            if not self.on_ground:
                self.on_ground = True
                self.squash_effect = 1.0
                # sfx: land_sound()
                self._create_particles(self.robot_pos + pygame.Vector2(0, self.robot_size.y/2), 15, self.COLOR_GROUND, (-3, 3), (-3, -1))

    def _update_world_scroll(self):
        if self.steps > 0 and self.steps % 500 == 0:
             self.world_scroll_speed += 0.05
        self.world_offset_x += self.world_scroll_speed

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _calculate_reward(self):
        # This logic is integrated directly into step() for clarity
        pass

    def _check_termination(self):
        if self.world_offset_x + self.robot_pos.x >= self.WORLD_LENGTH: return True
        if self.lives <= 0: return True
        if self.steps >= self.MAX_STEPS: return True
        return False
    
    def _handle_collisions(self):
        reward = 0
        robot_rect = self._get_robot_rect()

        for coin in self.coins[:]:
            coin_pos_on_screen = (coin['x'] - self.world_offset_x, coin['y'])
            if robot_rect.colliderect(pygame.Rect(coin_pos_on_screen[0] - coin['radius'], coin_pos_on_screen[1] - coin['radius'], coin['radius']*2, coin['radius']*2)):
                self.coins.remove(coin)
                self.score += 10
                reward += 1
                # sfx: coin_collect_sound()
                self._create_particles(pygame.Vector2(coin_pos_on_screen), 20, self.COLOR_COIN, (-4, 4), (-4, 4))
                break

        if self.invincibility_frames == 0:
            for obstacle in self.obstacles:
                obs_rect_on_screen = pygame.Rect(obstacle['x'] - self.world_offset_x, obstacle['y'], obstacle['w'], obstacle['h'])
                if robot_rect.colliderect(obs_rect_on_screen):
                    self.lives -= 1
                    reward -= 5
                    self.invincibility_frames = 60
                    self.squash_effect = 1.0
                    # sfx: hit_obstacle_sound()
                    self._create_particles(robot_rect.center, 30, self.COLOR_OBSTACLE, (-5, 5), (-5, 5))
                    break
        return reward
    
    def _get_robot_rect(self):
        squash = self.squash_effect
        w = self.robot_size.x * (1 + squash * 0.2 if self.on_ground else 1 - squash * 0.2)
        h = self.robot_size.y * (1 - squash * 0.2 if self.on_ground else 1 + squash * 0.2)
        x = self.robot_pos.x - w / 2
        y = self.robot_pos.y - h + self.robot_size.y / 2
        return pygame.Rect(x, y, w, h)

    def _generate_world(self):
        initial_chunks = math.ceil(self.SCREEN_WIDTH / self.CHUNK_SIZE) + 2
        for i in range(initial_chunks):
            self._generate_chunk(i * self.CHUNK_SIZE)

    def _manage_world_generation(self):
        current_chunk_idx = int((self.world_offset_x + self.SCREEN_WIDTH) / self.CHUNK_SIZE)
        if current_chunk_idx * self.CHUNK_SIZE < self.WORLD_LENGTH and current_chunk_idx not in self.generated_chunks:
            self._generate_chunk(current_chunk_idx * self.CHUNK_SIZE)

        despawn_x = self.world_offset_x - self.SCREEN_WIDTH
        self.obstacles = [o for o in self.obstacles if o['x'] + o['w'] > despawn_x]
        self.coins = [c for c in self.coins if c['x'] + c['radius'] > despawn_x]

    def _generate_chunk(self, chunk_start_x):
        if chunk_start_x in self.generated_chunks or chunk_start_x == 0: return
        self.generated_chunks.add(int(chunk_start_x / self.CHUNK_SIZE))

        num_obstacles = self.np_random.integers(2, 5)
        for _ in range(num_obstacles):
            w, h = self.np_random.integers(30, 80), self.np_random.integers(40, 120)
            x = chunk_start_x + self.np_random.integers(0, self.CHUNK_SIZE - w)
            self.obstacles.append({'x': x, 'y': self.GROUND_Y - h, 'w': w, 'h': h})

        num_coins = self.np_random.integers(3, 8)
        for _ in range(num_coins):
            x = chunk_start_x + self.np_random.integers(0, self.CHUNK_SIZE)
            y = self.GROUND_Y - self.np_random.integers(40, 150)
            self.coins.append({'x': x, 'y': y, 'radius': 12})
    
    def _create_particles(self, pos, count, color, x_vel_range, y_vel_range):
        for _ in range(count):
            self.particles.append({'pos': pygame.Vector2(pos), 'vel': pygame.Vector2(self.np_random.uniform(*x_vel_range), self.np_random.uniform(*y_vel_range)),
                'lifespan': self.np_random.integers(10, 25), 'color': color, 'size': self.np_random.integers(2, 5)})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        self._render_world_objects()
        self._render_robot()
        self._render_particles()

    def _render_background(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))
        for i in range(3):
            speed_factor = 0.1 * (i + 1)
            layer_offset = (self.world_offset_x * speed_factor) % self.SCREEN_WIDTH
            color = tuple(c * (0.6 + i*0.1) for c in self.COLOR_BG)
            for j in range(-1, 2):
                # Use a consistent seed for parallax layers to prevent flickering
                rng = random.Random((i+1) * 100 + j)
                for k in range(10):
                    x = rng.randint(0, self.SCREEN_WIDTH) + j * self.SCREEN_WIDTH - layer_offset
                    w, h = rng.randint(20, 80), rng.randint(50, 200)
                    pygame.draw.rect(self.screen, color, (x, self.GROUND_Y - h, w, h))

    def _render_world_objects(self):
        for obs in self.obstacles:
            rect = pygame.Rect(obs['x'] - self.world_offset_x, obs['y'], obs['w'], obs['h'])
            if rect.right > 0 and rect.left < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
        
        for coin in self.coins:
            x, y = int(coin['x'] - self.world_offset_x), int(coin['y'])
            if x + coin['radius'] > 0 and x - coin['radius'] < self.SCREEN_WIDTH:
                pygame.gfxdraw.filled_circle(self.screen, x, y, coin['radius'], self.COLOR_COIN)
                pygame.gfxdraw.aacircle(self.screen, x, y, coin['radius'], self.COLOR_COIN)
                shine_x = int(x + (math.sin(self.steps * 0.1) * coin['radius'] * 0.5))
                pygame.gfxdraw.filled_circle(self.screen, shine_x, y, int(coin['radius'] * 0.4), self.COLOR_COIN_SHINE)

        finish_x = self.WORLD_LENGTH - self.world_offset_x
        if finish_x < self.SCREEN_WIDTH:
            for i in range(10):
                color = self.COLOR_FINISH if i % 2 == 0 else (255, 255, 255)
                pygame.draw.rect(self.screen, color, (finish_x, i * self.SCREEN_HEIGHT/10, 10, self.SCREEN_HEIGHT/10))

    def _render_robot(self):
        robot_rect = self._get_robot_rect()
        color = (255, 255, 255) if self.invincibility_frames > 0 and self.steps % 10 < 5 else self.COLOR_ROBOT
        pygame.draw.rect(self.screen, color, robot_rect, border_radius=5)
        eye_x, eye_y = robot_rect.centerx + 5, robot_rect.centery - 10
        pygame.draw.rect(self.screen, self.COLOR_ROBOT_EYE, (eye_x, eye_y, 8, 8), border_radius=2)
    
    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            alpha = int(255 * (p['lifespan'] / 25.0))
            if alpha > 0:
                temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, p['color'] + (alpha,), (p['size'], p['size']), p['size'])
                self.screen.blit(temp_surf, (pos[0] - p['size'], pos[1] - p['size']))

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, (self.MAX_STEPS - self.steps) / 30)
        time_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        life_icon_surf = pygame.Surface((15, 25), pygame.SRCALPHA)
        pygame.draw.rect(life_icon_surf, self.COLOR_ROBOT, (0, 0, 15, 25), border_radius=3)
        for i in range(self.lives):
            self.screen.blit(life_icon_surf, (10 + i * 20, 40))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "progress": (self.world_offset_x + self.robot_pos.x) / self.WORLD_LENGTH
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows for manual testing of the environment
    env = GameEnv()
    obs, info = env.reset(seed=42)
    terminated = False
    total_reward = 0
    
    print("Starting manual play simulation (headless)...")
    print(f"User Guide: {env.user_guide}")
    
    for step_num in range(1, env.MAX_STEPS + 100):
        # A simple agent: jump if an obstacle is near
        action = env.action_space.sample() # Random action
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step_num % 100 == 0:
            print(f"Step {step_num}: Info={info}, Reward this step={reward:.2f}")

        if terminated or truncated:
            print("-" * 30)
            print(f"EPISODE FINISHED after {step_num} steps.")
            print(f"Final Info: {info}")
            print(f"Total Episode Reward: {total_reward:.2f}")
            print("-" * 30)
            
            # Reset for the next episode
            obs, info = env.reset(seed=step_num)
            total_reward = 0
            print("Environment reset. Starting new episode.")
            
    env.close()