import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import os
import math
import os
import pygame


# Set the environment variable to run Pygame in headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import pygame.gfxdraw


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a side-scrolling snail racing game.
    The player controls a snail, avoids obstacles, and collects boosts to reach the finish line.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: ↑/↓ to move the snail vertically. Press Space to activate a speed boost when you have one."
    )
    game_description = (
        "Control a snail in a side-view racing game, dodging obstacles to reach the finish line as fast as possible."
    )

    # Frame advance setting
    auto_advance = True

    # --- Game Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG_SKY = (135, 206, 235)
    COLOR_BG_HILLS = (34, 139, 34)
    COLOR_TRACK_GRASS = (50, 205, 50)
    COLOR_TRACK_DIRT = (139, 69, 19)
    COLOR_SNAIL_BODY = (255, 255, 102) # Bright Yellow
    COLOR_SNAIL_SHELL = (210, 105, 30)
    COLOR_OBSTACLE = (139, 69, 19)
    COLOR_BOOST_ITEM = (0, 191, 255)
    COLOR_FINISH_LINE_1 = (255, 255, 255)
    COLOR_FINISH_LINE_2 = (0, 0, 0)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (0, 0, 0)

    # World
    TRACK_LENGTH_PIXELS = 25000
    TRACK_Y_POS = 300
    TRACK_THICKNESS = 100

    # Snail
    SNAIL_X_POS = 100
    SNAIL_START_Y = TRACK_Y_POS - 20
    SNAIL_RADIUS = 15
    SNAIL_SHELL_RADIUS = 12
    SNAIL_V_SPEED = 8
    SNAIL_BASE_H_SPEED = 5
    SNAIL_BOOST_H_SPEED = 15
    SNAIL_COLLISION_SLOWDOWN_FRAMES = 15

    # Obstacles
    OBSTACLE_WIDTH = 30
    OBSTACLE_HEIGHT = 40
    OBSTACLE_MIN_SPEED = -4
    OBSTACLE_MAX_SPEED = -8
    OBSTACLE_SPAWN_INTERVAL = 120 # pixels of world travel

    # Boosts
    BOOST_RADIUS = 12
    BOOST_SPAWN_INTERVAL = 400
    BOOST_DURATION_FRAMES = 90 # 3 seconds

    # Game Rules
    MAX_COLLISIONS = 4
    TIME_LIMIT_SECONDS = 90.0
    MAX_STEPS = int(TIME_LIMIT_SECONDS * FPS * 1.1) # A bit more than time limit allows

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Internal state variables are initialized in reset()
        self.snail_y = 0
        self.snail_vy = 0
        self.world_x = 0
        self.game_time_seconds = 0
        self.collisions = 0
        self.boost_charges = 0
        self.boost_active_timer = 0
        self.collision_slowdown_timer = 0
        self.obstacles = []
        self.boost_items = []
        self.particles = []
        self.next_obstacle_spawn_x = 0
        self.next_boost_spawn_x = 0
        self.difficulty_timer = 0
        self.current_obstacle_speed_factor = 1.0
        self.rng = None
        self.score = 0
        self.steps = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.world_x = 0
        self.game_time_seconds = 0
        self.collisions = 0
        self.difficulty_timer = 0
        self.current_obstacle_speed_factor = 1.0

        # Snail state
        self.snail_y = self.SNAIL_START_Y
        self.snail_vy = 0
        self.boost_charges = 1 # Start with one boost
        self.boost_active_timer = 0
        self.collision_slowdown_timer = 0

        # Entity lists
        self.obstacles = []
        self.boost_items = []
        self.particles = []
        
        # Spawners
        self.next_obstacle_spawn_x = self.SCREEN_WIDTH * 2
        self.next_boost_spawn_x = self.SCREEN_WIDTH * 1.5

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        reward = 0
        
        # 1. Handle Input
        self._handle_input(action)
        
        # 2. Update Game State
        self._update_world()
        self._update_snail()
        self._update_entities()
        self._update_particles()
        
        # 3. Check Collisions and apply rewards/penalties
        reward += self._check_collisions()

        # 4. Continuous Rewards
        reward += 0.01  # Small reward for surviving
        if self.boost_charges > 0 and not (action[1] == 1):
             reward -= 0.02 # Penalty for not using available boost

        # 5. Check Termination Conditions
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.world_x >= self.TRACK_LENGTH_PIXELS:
                time_bonus = max(0, self.TIME_LIMIT_SECONDS - self.game_time_seconds)
                reward += 50 + time_bonus # Win bonus
            else:
                reward -= 50 # Lose penalty
        
        self.score += reward
        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1

        # Vertical movement
        if movement == 1: # Up
            self.snail_vy = -self.SNAIL_V_SPEED
        elif movement == 2: # Down
            self.snail_vy = self.SNAIL_V_SPEED
        else:
            self.snail_vy = 0

        # Activate boost
        if space_pressed and self.boost_charges > 0 and self.boost_active_timer <= 0:
            self.boost_charges -= 1
            self.boost_active_timer = self.BOOST_DURATION_FRAMES
            # Sound effect placeholder: # SFX_BOOST_ACTIVATE
            self._create_particles(
                count=30,
                pos=(self.SNAIL_X_POS, self.snail_y),
                min_vel=(-2, -3), max_vel=(2, 3),
                min_lifespan=15, max_lifespan=30,
                color=(255, 255, 0)
            )


    def _update_world(self):
        # Update timers
        self.game_time_seconds += 1 / self.FPS
        self.difficulty_timer += 1 / self.FPS
        
        if self.boost_active_timer > 0:
            self.boost_active_timer -= 1
        
        if self.collision_slowdown_timer > 0:
            self.collision_slowdown_timer -= 1
        
        # Calculate snail's horizontal speed
        if self.collision_slowdown_timer > 0:
            current_h_speed = self.SNAIL_BASE_H_SPEED / 2
        elif self.boost_active_timer > 0:
            current_h_speed = self.SNAIL_BOOST_H_SPEED
        else:
            current_h_speed = self.SNAIL_BASE_H_SPEED

        # Scroll the world
        self.world_x += current_h_speed

        # Difficulty scaling
        if self.difficulty_timer >= 10:
            self.difficulty_timer = 0
            self.current_obstacle_speed_factor += 0.05
    
    def _update_snail(self):
        self.snail_y += self.snail_vy
        # Clamp snail position to the track area
        track_top = self.TRACK_Y_POS - self.SNAIL_RADIUS
        track_bottom = self.TRACK_Y_POS + self.TRACK_THICKNESS - self.SNAIL_RADIUS
        self.snail_y = np.clip(self.snail_y, track_top, track_bottom)

    def _update_entities(self):
        # Update obstacles
        for obstacle in self.obstacles:
            obstacle['x'] += obstacle['vx']
        self.obstacles = [o for o in self.obstacles if o['x'] + self.OBSTACLE_WIDTH > 0]

        # Update boost items (they are static relative to the world)
        self.boost_items = [b for b in self.boost_items if b['x'] - self.world_x + self.BOOST_RADIUS > 0]

        # Spawn new obstacles
        if self.world_x > self.next_obstacle_spawn_x:
            self._spawn_obstacle()
            self.next_obstacle_spawn_x += self.OBSTACLE_SPAWN_INTERVAL * self.rng.uniform(0.8, 1.5)

        # Spawn new boost items
        if self.world_x > self.next_boost_spawn_x:
            self._spawn_boost_item()
            self.next_boost_spawn_x += self.BOOST_SPAWN_INTERVAL * self.rng.uniform(0.9, 1.8)

    def _spawn_obstacle(self):
        y_pos = self.rng.integers(
            self.TRACK_Y_POS,
            self.TRACK_Y_POS + self.TRACK_THICKNESS - self.OBSTACLE_HEIGHT
        )
        # FIX: Swapped MIN and MAX speed as MIN(-4) was > MAX(-8), causing an error.
        speed = self.rng.uniform(self.OBSTACLE_MAX_SPEED, self.OBSTACLE_MIN_SPEED) * self.current_obstacle_speed_factor
        
        self.obstacles.append({
            'x': self.world_x + self.SCREEN_WIDTH,
            'y': y_pos,
            'vx': speed
        })

    def _spawn_boost_item(self):
        y_pos = self.rng.integers(
            self.TRACK_Y_POS,
            self.TRACK_Y_POS + self.TRACK_THICKNESS - self.BOOST_RADIUS * 2
        )
        self.boost_items.append({
            'x': self.world_x + self.SCREEN_WIDTH,
            'y': y_pos,
        })
    
    def _update_particles(self):
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _check_collisions(self):
        reward = 0
        snail_rect = pygame.Rect(
            self.SNAIL_X_POS - self.SNAIL_RADIUS,
            self.snail_y - self.SNAIL_RADIUS,
            self.SNAIL_RADIUS * 2,
            self.SNAIL_RADIUS * 2
        )
        
        # Obstacles
        for obstacle in self.obstacles[:]:
            obstacle_rect = pygame.Rect(
                obstacle['x'] - self.world_x,
                obstacle['y'],
                self.OBSTACLE_WIDTH,
                self.OBSTACLE_HEIGHT
            )
            if snail_rect.colliderect(obstacle_rect):
                self.obstacles.remove(obstacle)
                self.collisions += 1
                self.collision_slowdown_timer = self.SNAIL_COLLISION_SLOWDOWN_FRAMES
                reward -= 10
                # Sound effect placeholder: # SFX_COLLISION
                self._create_particles(
                    count=20,
                    pos=(self.SNAIL_X_POS, self.snail_y),
                    min_vel=(-3, -3), max_vel=(3, 0),
                    min_lifespan=20, max_lifespan=40,
                    color=self.COLOR_OBSTACLE
                )
                break
        
        # Boost items
        for boost in self.boost_items[:]:
            boost_rect = pygame.Rect(
                boost['x'] - self.world_x - self.BOOST_RADIUS,
                boost['y'] - self.BOOST_RADIUS,
                self.BOOST_RADIUS * 2,
                self.BOOST_RADIUS * 2
            )
            if snail_rect.colliderect(boost_rect):
                self.boost_items.remove(boost)
                self.boost_charges = min(3, self.boost_charges + 1)
                reward += 5
                # Sound effect placeholder: # SFX_BOOST_COLLECT
                self._create_particles(
                    count=15,
                    pos=(boost_rect.centerx, boost_rect.centery),
                    min_vel=(-2, -2), max_vel=(2, 2),
                    min_lifespan=10, max_lifespan=20,
                    color=self.COLOR_BOOST_ITEM
                )
                break
        
        return reward

    def _check_termination(self):
        return (
            self.collisions >= self.MAX_COLLISIONS or
            self.game_time_seconds >= self.TIME_LIMIT_SECONDS or
            self.world_x >= self.TRACK_LENGTH_PIXELS
        )

    def _get_observation(self):
        self._render_all()
        # Pygame surfaces are (width, height), but our observation space is (height, width)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_seconds": self.game_time_seconds,
            "collisions": self.collisions,
            "progress_percent": (self.world_x / self.TRACK_LENGTH_PIXELS) * 100,
            "boost_charges": self.boost_charges,
        }

    def _create_particles(self, count, pos, min_vel, max_vel, min_lifespan, max_lifespan, color):
        for _ in range(count):
            self.particles.append({
                'x': pos[0], 'y': pos[1],
                'vx': self.rng.uniform(min_vel[0], max_vel[0]),
                'vy': self.rng.uniform(min_vel[1], max_vel[1]),
                'lifespan': self.rng.integers(min_lifespan, max_lifespan),
                'color': color,
                'size': self.rng.integers(2, 5)
            })

    # --- Rendering Methods ---

    def _render_all(self):
        self._render_background()
        self._render_track()
        self._render_entities()
        self._render_particles()
        self._render_ui()

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_SKY)
        # Parallax hills
        hill_x_offset = (self.world_x * 0.2) % self.SCREEN_WIDTH
        for i in range(-1, 2):
            pygame.draw.circle(self.screen, self.COLOR_BG_HILLS, (i * self.SCREEN_WIDTH + hill_x_offset, 350), 200)

    def _render_track(self):
        # Grass base
        pygame.draw.rect(self.screen, self.COLOR_TRACK_GRASS, (0, self.TRACK_Y_POS, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.TRACK_Y_POS))
        # Dirt track
        pygame.draw.rect(self.screen, self.COLOR_TRACK_DIRT, (0, self.TRACK_Y_POS, self.SCREEN_WIDTH, self.TRACK_THICKNESS))
        
        # Finish line
        finish_x_on_screen = self.TRACK_LENGTH_PIXELS - self.world_x
        if finish_x_on_screen < self.SCREEN_WIDTH:
            check_size = 20
            for i in range(int(self.TRACK_THICKNESS / check_size)):
                for j in range(2):
                    color = self.COLOR_FINISH_LINE_1 if (i + j) % 2 == 0 else self.COLOR_FINISH_LINE_2
                    pygame.draw.rect(self.screen, color, (
                        finish_x_on_screen + j * check_size,
                        self.TRACK_Y_POS + i * check_size,
                        check_size, check_size
                    ))

    def _render_entities(self):
        # Obstacles
        for o in self.obstacles:
            screen_x = o['x'] - self.world_x
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (screen_x, o['y'], self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT))
            pygame.draw.rect(self.screen, (0,0,0), (screen_x, o['y'], self.OBSTACLE_WIDTH, self.OBSTACLE_HEIGHT), 2)

        # Boost items
        for b in self.boost_items:
            screen_x = b['x'] - self.world_x
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
            radius = int(self.BOOST_RADIUS * (1 + pulse * 0.2))
            pygame.gfxdraw.filled_circle(self.screen, int(screen_x), int(b['y']), radius, self.COLOR_BOOST_ITEM)
            pygame.gfxdraw.aacircle(self.screen, int(screen_x), int(b['y']), radius, self.COLOR_BOOST_ITEM)

        # Snail
        self._render_snail()

    def _render_snail(self):
        x, y = int(self.SNAIL_X_POS), int(self.snail_y)
        
        # Boost trail
        if self.boost_active_timer > 0:
            for i in range(10):
                trail_x = x - i * 4
                trail_y = y
                alpha = 200 - i * 20
                color = (*self.COLOR_SNAIL_BODY, alpha)
                s = pygame.Surface((self.SNAIL_RADIUS, self.SNAIL_RADIUS), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (self.SNAIL_RADIUS // 2, self.SNAIL_RADIUS // 2), self.SNAIL_RADIUS // 2 - i)
                self.screen.blit(s, (trail_x - self.SNAIL_RADIUS // 2, trail_y - self.SNAIL_RADIUS // 2))

        # Snail body
        body_color = (255, 255, 255) if self.boost_active_timer > 0 else self.COLOR_SNAIL_BODY
        pygame.draw.ellipse(self.screen, body_color, (x - self.SNAIL_RADIUS, y - self.SNAIL_RADIUS // 2, self.SNAIL_RADIUS * 1.8, self.SNAIL_RADIUS))
        
        # Snail shell with wobble
        wobble = math.sin(self.world_x * 0.1) * 2
        pygame.draw.circle(self.screen, self.COLOR_SNAIL_SHELL, (x, y - 10 + int(wobble)), self.SNAIL_SHELL_RADIUS)
        pygame.draw.circle(self.screen, (0,0,0), (x, y - 10 + int(wobble)), self.SNAIL_SHELL_RADIUS, 2)

        # Eye
        pygame.draw.circle(self.screen, (255, 255, 255), (x + 10, y - 8), 4)
        pygame.draw.circle(self.screen, (0, 0, 0), (x + 11, y - 8), 2)

    def _render_particles(self):
        for p in self.particles:
            lifespan_ratio = p['lifespan'] / self.rng.integers(p.get('min_lifespan', 10), p.get('max_lifespan', 20))
            size = int(p['size'] * lifespan_ratio)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), size)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow_surf = font.render(text, True, self.COLOR_UI_SHADOW)
            text_surf = font.render(text, True, color)
            self.screen.blit(shadow_surf, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surf, pos)

        # Collisions
        collision_text = f"Collisions: {self.collisions}/{self.MAX_COLLISIONS}"
        draw_text(collision_text, self.font_small, self.COLOR_UI_TEXT, (10, 10))

        # Timer
        time_left = max(0, self.TIME_LIMIT_SECONDS - self.game_time_seconds)
        timer_text = f"Time: {time_left:.1f}s"
        text_width = self.font_small.size(timer_text)[0]
        draw_text(timer_text, self.font_small, self.COLOR_UI_TEXT, (self.SCREEN_WIDTH - text_width - 10, 10))
        
        # Boost charges
        boost_text = f"Boosts: {self.boost_charges}"
        text_width = self.font_small.size(boost_text)[0]
        draw_text(boost_text, self.font_small, self.COLOR_UI_TEXT, (self.SCREEN_WIDTH // 2 - text_width // 2, self.SCREEN_HEIGHT - 40))

        # Progress bar
        progress_ratio = self.world_x / self.TRACK_LENGTH_PIXELS
        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 10
        pygame.draw.rect(self.screen, (50,50,50), (10, self.SCREEN_HEIGHT - 20, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_BOOST_ITEM, (10, self.SCREEN_HEIGHT - 20, bar_width * progress_ratio, bar_height))
        pygame.draw.rect(self.screen, (255,255,255), (10, self.SCREEN_HEIGHT - 20, bar_width, bar_height), 1)

        # Game Over Message
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,128))
            self.screen.blit(s, (0,0))
            if self.world_x >= self.TRACK_LENGTH_PIXELS:
                msg = "FINISH!"
            else:
                msg = "GAME OVER"
            text_width, text_height = self.font_large.size(msg)
            draw_text(msg, self.font_large, (255,215,0), ((self.SCREEN_WIDTH - text_width)//2, (self.SCREEN_HEIGHT - text_height)//2))

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    import time

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part is for human testing and requires a display.
    # It will not run in a headless environment.
    is_headless = True
    try:
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Snail Racer")
        is_headless = False
    except pygame.error:
        print("Pygame display unavailable. Running in headless mode.")

    if is_headless:
        # Run a simple test episode if headless
        print("Running a headless test episode...")
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0
        start_time = time.time()
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
        end_time = time.time()
        print(f"Episode finished in {step_count} steps ({end_time - start_time:.2f}s).")
        print(f"Final Info: {info}")
    else:
        # Interactive mode
        obs, info = env.reset()
        done = False
        
        # Key states
        keys_held = {
            pygame.K_UP: 0, pygame.K_DOWN: 0,
            pygame.K_SPACE: 0, pygame.K_LSHIFT: 0
        }

        running = True
        while running:
            # --- Action Mapping for Human Play ---
            movement = 0
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in keys_held: keys_held[event.key] = 1
                    if event.key == pygame.K_r: # Reset on 'r'
                        obs, info = env.reset()
                        done = False
                if event.type == pygame.KEYUP:
                    if event.key in keys_held: keys_held[event.key] = 0

            if keys_held[pygame.K_UP]: movement = 1
            elif keys_held[pygame.K_DOWN]: movement = 2
            
            space = keys_held[pygame.K_SPACE]
            shift = keys_held[pygame.K_LSHIFT]

            action = [movement, space, shift]

            # --- Step the Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- Render to screen ---
            frame_to_show = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame_to_show)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if done:
                print(f"Episode finished. Score: {info['score']:.2f}, Time: {info['time_seconds']:.2f}s")
                time.sleep(2) # Pause before resetting
                obs, info = env.reset()

            env.clock.tick(env.FPS)
        
    env.close()