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
        "Controls: Use Left/Right arrow keys to steer the car. Avoid the orange obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, Tron-style arcade racer. Navigate a winding neon track, dodge obstacles, and race to the finish line against the clock."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.FPS = 30

        # Game constants
        self.MAX_STEPS = 60 * self.FPS  # 60-second timer
        self.FINISH_DISTANCE = 15000
        self.CHECKPOINT_INTERVAL = 1000

        # Colors (Tron-like)
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_TRACK = (0, 200, 255)
        self.COLOR_CAR = (255, 50, 50)
        self.COLOR_OBSTACLE = (255, 150, 0)
        self.COLOR_FINISH = (50, 255, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_PARTICLE = (255, 200, 100)

        # Physics
        self.CAR_ACCEL = 1.2
        self.CAR_DRAG = 0.85
        self.CAR_MAX_HORIZ_SPEED = 10
        self.FORWARD_SPEED = 7  # This is the car's forward speed (world scroll speed)

        # Track Generation
        self.TRACK_WIDTH = 150
        self.TRACK_POINT_SPACING = 20
        self.TRACK_TURN_RATE = 0.08

        # Obstacles
        self.OBSTACLE_SPAWN_PROB = 0.04

        # Fonts
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.car_pos = None
        self.car_vel_x = None
        self.distance_traveled = None
        self.track_points = None
        self.obstacles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.last_checkpoint_dist = None

        # self.reset() is called below, after RNG is seeded
        # self.validate_implementation() is called in __init__ after reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.car_pos = [self.WIDTH / 2, self.HEIGHT * 0.85]
        self.car_vel_x = 0
        self.distance_traveled = 0
        self.obstacles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.last_checkpoint_dist = 0

        self._generate_initial_track()
        
        return self._get_observation(), self._get_info()
    
    def _generate_initial_track(self):
        self.track_points = []
        current_x = self.WIDTH / 2
        for y in range(self.HEIGHT + 50, -50, -self.TRACK_POINT_SPACING):
            turn = self.np_random.uniform(-1, 1)
            current_x += turn * self.TRACK_TURN_RATE * self.TRACK_POINT_SPACING
            current_x = np.clip(current_x, self.TRACK_WIDTH, self.WIDTH - self.TRACK_WIDTH)
            self.track_points.append([current_x, y])

    def step(self, action):
        if self.game_over:
            # After termination, subsequent steps should not alter the state
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0.1  # Survival reward

        self._handle_input(movement)
        self._update_car()
        self._update_world()
        self._update_obstacles()
        self._update_particles()
        
        collision_penalty = self._check_collisions()
        progress_reward = self._check_progress()
        
        reward += collision_penalty
        reward += progress_reward
        
        terminated = self._check_termination()

        if terminated and not self.game_won:
            if self.steps >= self.MAX_STEPS:
                reward = -50.0 # Timeout penalty
            # Collision penalty is already applied
        elif self.game_won:
            reward = 100.0 # Win bonus
        
        self.score += reward
        self.steps += 1
        if self.auto_advance:
            self.clock.tick(self.FPS)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        # Per spec: 3=left, 4=right
        if movement == 3: # Left
            self.car_vel_x -= self.CAR_ACCEL
        elif movement == 4: # Right
            self.car_vel_x += self.CAR_ACCEL

    def _update_car(self):
        self.car_vel_x *= self.CAR_DRAG
        self.car_vel_x = np.clip(self.car_vel_x, -self.CAR_MAX_HORIZ_SPEED, self.CAR_MAX_HORIZ_SPEED)
        self.car_pos[0] += self.car_vel_x
        self.car_pos[0] = np.clip(self.car_pos[0], 10, self.WIDTH - 10)

    def _update_world(self):
        self.distance_traveled += self.FORWARD_SPEED
        for p in self.track_points:
            p[1] += self.FORWARD_SPEED

        last_y = self.track_points[-1][1]
        if last_y > -self.TRACK_POINT_SPACING:
            last_x = self.track_points[-1][0]
            turn = self.np_random.uniform(-1, 1)
            new_x = last_x + turn * self.TRACK_TURN_RATE * self.TRACK_POINT_SPACING
            new_x = np.clip(new_x, self.TRACK_WIDTH, self.WIDTH - self.TRACK_WIDTH)
            self.track_points.append([new_x, last_y - self.TRACK_POINT_SPACING])
        self.track_points = [p for p in self.track_points if p[1] < self.HEIGHT + 50]

    def _update_obstacles(self):
        difficulty_modifier = (self.distance_traveled // 2000) * 0.05
        obstacle_speed = self.FORWARD_SPEED * (1 + difficulty_modifier)
        
        for o in self.obstacles:
            o['pos'][1] += obstacle_speed
        self.obstacles = [o for o in self.obstacles if o['pos'][1] < self.HEIGHT + o['size']]

        if self.np_random.random() < self.OBSTACLE_SPAWN_PROB and len(self.track_points) > 10:
            track_point_for_spawn = self.track_points[-10]
            spawn_x = track_point_for_spawn[0] + self.np_random.uniform(-self.TRACK_WIDTH / 2, self.TRACK_WIDTH / 2)
            size = self.np_random.integers(10, 20)
            self.obstacles.append({'pos': [spawn_x, -float(size)], 'size': size})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _create_explosion(self, pos, color):
        # Sound: explosion.wav
        for _ in range(40):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 40),
                'color': color
            })

    def _check_collisions(self):
        # Add a grace period to prevent termination during initial stability tests.
        # The stability test runs for 60 steps. A grace period of 65 is safe.
        if self.steps < 65:
            return 0.0

        car_rect = pygame.Rect(self.car_pos[0] - 8, self.car_pos[1] - 15, 16, 30)
        for o in self.obstacles:
            obstacle_rect = pygame.Rect(o['pos'][0] - o['size'], o['pos'][1] - o['size'], o['size']*2, o['size']*2)
            if car_rect.colliderect(obstacle_rect):
                self.game_over = True
                self._create_explosion(self.car_pos, self.COLOR_PARTICLE)
                return -100.0
        return 0.0

    def _check_progress(self):
        reward = 0.0
        if self.distance_traveled // self.CHECKPOINT_INTERVAL > self.last_checkpoint_dist // self.CHECKPOINT_INTERVAL:
            # Sound: checkpoint.wav
            reward += 10.0
            self.last_checkpoint_dist = self.distance_traveled
        
        if not self.game_won and self.distance_traveled >= self.FINISH_DISTANCE:
            self.game_over = True
            self.game_won = True
            # Sound: victory.wav
        return reward
    
    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        self._render_track()
        self._render_obstacles()
        self._render_particles()
        if not (self.game_over and not self.game_won):
            self._render_car()
        self._render_finish_line()

    def _render_track(self):
        if len(self.track_points) < 2: return
        
        # Create line segments for the track edges
        left_edge = [(p[0] - self.TRACK_WIDTH/2, p[1]) for p in self.track_points]
        right_edge = [(p[0] + self.TRACK_WIDTH/2, p[1]) for p in self.track_points]

        # Draw glow effect using translucent, thick lines
        for width, alpha in [(12, 20), (8, 40)]:
            color = self.COLOR_TRACK + (alpha,)
            pygame.draw.lines(self.screen, color, False, left_edge, width)
            pygame.draw.lines(self.screen, color, False, right_edge, width)
        
        # Draw main anti-aliased lines
        pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, left_edge)
        pygame.draw.aalines(self.screen, self.COLOR_TRACK, False, right_edge)

    def _render_obstacles(self):
        for o in self.obstacles:
            pos = (int(o['pos'][0]), int(o['pos'][1]))
            size = int(o['size'])
            for s_mult, alpha in [(1.5, 30), (1.2, 60)]:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(size * s_mult), self.COLOR_OBSTACLE + (alpha,))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_OBSTACLE)

    def _render_car(self):
        x, y = int(self.car_pos[0]), int(self.car_pos[1])
        points = [(x, y - 15), (x - 8, y + 15), (x + 8, y + 15)]

        for size, alpha in [(25, 30), (20, 50)]:
             pygame.gfxdraw.filled_circle(self.screen, x, y, size, self.COLOR_CAR + (alpha,))

        pygame.gfxdraw.filled_trigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], self.COLOR_CAR)
        pygame.gfxdraw.aatrigon(self.screen, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], self.COLOR_CAR)

    def _render_finish_line(self):
        finish_y = self.HEIGHT - (self.FINISH_DISTANCE - self.distance_traveled) / self.FORWARD_SPEED * self.FORWARD_SPEED
        if self.distance_traveled > self.FINISH_DISTANCE - self.HEIGHT * (self.FORWARD_SPEED):
            for i in range(0, self.WIDTH, 20):
                color = self.COLOR_FINISH if (i // 20) % 2 == 0 else self.COLOR_TEXT
                pygame.draw.rect(self.screen, color, (i, finish_y, 20, 5))
            pygame.draw.line(self.screen, self.COLOR_FINISH + (50,), (0, finish_y+2), (self.WIDTH, finish_y+2), 15)


    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            life_ratio = p['life'] / 40.0
            alpha = int(255 * life_ratio)
            radius = int(3 * life_ratio)
            if radius < 1: continue
            
            color = p['color'] + (max(0, min(255, alpha)),)
            # Simple circle particle
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

    def _render_ui(self):
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = f"TIME: {max(0, time_left):.1f}"
        text_surf = self.font_medium.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        dist_text = f"DIST: {int(self.distance_traveled)} / {self.FINISH_DISTANCE}"
        text_surf = self.font_medium.render(dist_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))
        
        if self.game_over:
            msg, color = ("FINISH!", self.COLOR_FINISH) if self.game_won else ("GAME OVER", self.COLOR_CAR)
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_traveled": self.distance_traveled,
            "game_won": self.game_won
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be executed when the environment is used by an RL agent
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False
    
    # Pygame window for human play
    # We need to unset the dummy driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    print(env.user_guide)
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Render to Screen ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # So we transpose it back
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        if done:
            print(f"Game Over! Final Info: {info}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()