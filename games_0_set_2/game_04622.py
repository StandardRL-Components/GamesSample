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


class Particle:
    """A simple class for managing particle effects."""
    def __init__(self, pos, vel, color, lifespan):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.color = color
        self.lifespan = lifespan
        self.initial_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95  # Damping
        self.lifespan -= 1

    def draw(self, surface, iso_to_screen_fn, offset):
        if self.lifespan > 0:
            screen_pos = iso_to_screen_fn(self.pos.x, self.pos.y) + offset
            alpha = int(255 * (self.lifespan / self.initial_lifespan))
            alpha = max(0, min(255, alpha))
            size = max(1, int(3 * (self.lifespan / self.initial_lifespan)))
            
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (size, size), size)
            surface.blit(temp_surf, (int(screen_pos.x - size), int(screen_pos.y - size)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to aim, ↑↓ to adjust power. Press space to hit the ball."
    )

    game_description = (
        "A serene isometric mini-golf game. Sink the ball in 9 procedurally generated holes with the fewest strokes."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_SIZE = 250
        self.MAX_STROKES = 10
        self.NUM_HOLES = 9
        self.STOP_THRESHOLD = 0.1
        self.MAX_POWER = 100
        self.POWER_INCREMENT = 2
        self.ANGLE_INCREMENT = 3

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_FAIRWAY = (63, 132, 92)
        self.COLOR_ROUGH = (48, 102, 72)
        self.COLOR_OBSTACLE = (100, 110, 120)
        self.COLOR_OBSTACLE_BORDER = (120, 130, 140)
        self.COLOR_BALL = (230, 230, 230)
        self.COLOR_BALL_SHADOW = (0, 0, 0, 64)
        self.COLOR_HOLE = (30, 30, 30)
        self.COLOR_FLAG = (210, 60, 60)
        self.COLOR_TRAJECTORY = (255, 200, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_POWER_BAR_BG = (50, 50, 50)
        self.COLOR_POWER_BAR_FILL = (255, 180, 0)
        
        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        self.iso_offset = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT // 2 - 80)
        
        # Initialize state variables
        self.ball_pos = pygame.math.Vector2(0, 0)
        self.ball_vel = pygame.math.Vector2(0, 0)
        self.hole_pos = pygame.math.Vector2(0, 0)
        self.start_pos = pygame.math.Vector2(0, 0)
        self.obstacles = []
        self.particles = []

        # self.reset() is called by the environment wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.total_score = 0
        self.current_hole_num = 1
        self.game_over = False
        self._reset_hole()
        
        return self._get_observation(), self._get_info()

    def _reset_hole(self):
        self.strokes = 0
        self.game_phase = "aiming"
        self.aim_angle = 90
        self.power = 0
        self.ball_vel.update(0, 0)
        self.particles = []
        
        self._generate_hole()
        self.ball_pos.update(self.start_pos)

    def _generate_hole(self):
        self.obstacles = []
        
        while True:
            self.start_pos = pygame.math.Vector2(
                self.np_random.uniform(-self.WORLD_SIZE * 0.8, self.WORLD_SIZE * 0.8),
                self.np_random.uniform(-self.WORLD_SIZE * 0.8, self.WORLD_SIZE * 0.8),
            )
            self.hole_pos = pygame.math.Vector2(
                self.np_random.uniform(-self.WORLD_SIZE * 0.8, self.WORLD_SIZE * 0.8),
                self.np_random.uniform(-self.WORLD_SIZE * 0.8, self.WORLD_SIZE * 0.8),
            )
            if self.start_pos.distance_to(self.hole_pos) > self.WORLD_SIZE * 0.5:
                break

        num_obstacles = min(5, self.current_hole_num // 2 + self.np_random.integers(0, 2))
        for _ in range(num_obstacles):
            while True:
                pos = pygame.math.Vector2(
                    self.np_random.uniform(-self.WORLD_SIZE, self.WORLD_SIZE),
                    self.np_random.uniform(-self.WORLD_SIZE, self.WORLD_SIZE),
                )
                if pos.distance_to(self.start_pos) > 40 and pos.distance_to(self.hole_pos) > 40:
                    size = self.np_random.uniform(10, 25)
                    self.obstacles.append({'pos': pos, 'radius': size})
                    break
    
    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        truncated = False

        if self.game_phase == "aiming":
            self._handle_aiming(movement, space_held)
        elif self.game_phase == "ball_moving":
            reward += self._update_physics()

        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_aiming(self, movement, space_held):
        if movement == 1:  # Up
            self.power = min(self.MAX_POWER, self.power + self.POWER_INCREMENT)
        elif movement == 2:  # Down
            self.power = max(0, self.power - self.POWER_INCREMENT)
        elif movement == 3:  # Left
            self.aim_angle = (self.aim_angle - self.ANGLE_INCREMENT) % 360
        elif movement == 4:  # Right
            self.aim_angle = (self.aim_angle + self.ANGLE_INCREMENT) % 360

        if space_held and self.power > 0:
            # Hit the ball
            self.strokes += 1
            self.game_phase = "ball_moving"
            rad_angle = math.radians(self.aim_angle)
            force = self.power / self.MAX_POWER * 15
            self.ball_vel.x = math.cos(rad_angle) * force
            self.ball_vel.y = math.sin(rad_angle) * force
            # Sound: "golf_hit.wav"

    def _update_physics(self):
        reward = -0.1 # Small penalty for each physics step
        
        dist_before = self.ball_pos.distance_to(self.hole_pos)
        
        self.ball_pos += self.ball_vel
        friction = 0.97 if self._get_terrain(self.ball_pos) == 'fairway' else 0.94
        self.ball_vel *= friction

        if abs(self.ball_pos.x) > self.WORLD_SIZE:
            self.ball_pos.x = self.WORLD_SIZE * np.sign(self.ball_pos.x)
            self.ball_vel.x *= -0.8
            self._create_spark_effect(self.ball_pos)
        if abs(self.ball_pos.y) > self.WORLD_SIZE:
            self.ball_pos.y = self.WORLD_SIZE * np.sign(self.ball_pos.y)
            self.ball_vel.y *= -0.8
            self._create_spark_effect(self.ball_pos)

        for obs in self.obstacles:
            dist_vec = self.ball_pos - obs['pos']
            if dist_vec.length() < obs['radius'] + 5: # 5 is ball radius
                reward -= 2.0
                # Sound: "obstacle_hit.wav"
                self._create_spark_effect(self.ball_pos)
                
                overlap = (obs['radius'] + 5) - dist_vec.length()
                if dist_vec.length() > 0:
                    self.ball_pos += dist_vec.normalize() * overlap
                
                normal = dist_vec.normalize()
                self.ball_vel = self.ball_vel.reflect(normal) * 0.8
        
        if self.ball_pos.distance_to(self.hole_pos) < 7 and self.ball_vel.length() < 3.0:
            # Sound: "ball_in_hole.wav"
            reward += 5.0
            self.total_score += self.strokes
            self.current_hole_num += 1
            if self.current_hole_num > self.NUM_HOLES:
                reward += 50.0
                self.game_over = True
            else:
                self._reset_hole()
            return reward

        if self.ball_vel.length() < self.STOP_THRESHOLD:
            self.ball_vel.update(0, 0)
            self.game_phase = "aiming"
            self.power = 0
            
            dist_after = self.ball_pos.distance_to(self.hole_pos)
            reward += (dist_before - dist_after) * 0.1

            if self.strokes >= self.MAX_STROKES:
                reward -= 50.0
                self.game_over = True
        
        return reward

    def _get_terrain(self, pos):
        return 'fairway'

    def _iso_to_screen(self, x, y):
        return pygame.math.Vector2(
            (x - y) * 0.9,
            (x + y) * 0.5
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        pygame.gfxdraw.filled_polygon(self.screen, [
            self._iso_to_screen(-self.WORLD_SIZE, -self.WORLD_SIZE) + self.iso_offset,
            self._iso_to_screen(self.WORLD_SIZE, -self.WORLD_SIZE) + self.iso_offset,
            self._iso_to_screen(self.WORLD_SIZE, self.WORLD_SIZE) + self.iso_offset,
            self._iso_to_screen(-self.WORLD_SIZE, self.WORLD_SIZE) + self.iso_offset
        ], self.COLOR_FAIRWAY)

        render_list = []
        for obs in self.obstacles:
            render_list.append(('obstacle', obs, obs['pos'].y))
        render_list.append(('hole', self.hole_pos, self.hole_pos.y))
        if self.game_phase == "ball_moving" or self.power == 0:
            render_list.append(('ball', self.ball_pos, self.ball_pos.y))
        
        render_list.sort(key=lambda item: item[2])
        
        for item_type, data, _ in render_list:
            if item_type == 'obstacle':
                self._render_obstacle(data)
            elif item_type == 'hole':
                self._render_hole_and_flag(data)
            elif item_type == 'ball':
                self._render_ball(data)

        for p in self.particles:
            p.update()
            p.draw(self.screen, self._iso_to_screen, self.iso_offset)
        self.particles = [p for p in self.particles if p.lifespan > 0]
        
        if self.game_phase == "aiming":
            self._render_aiming_assists()
            self._render_ball(self.ball_pos)

    def _render_obstacle(self, obs):
        pos = self._iso_to_screen(obs['pos'].x, obs['pos'].y) + self.iso_offset
        radius = obs['radius'] * 0.7
        pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), self.COLOR_OBSTACLE)
        pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), self.COLOR_OBSTACLE_BORDER)

    def _render_hole_and_flag(self, pos):
        screen_pos = self._iso_to_screen(pos.x, pos.y) + self.iso_offset
        pygame.gfxdraw.filled_ellipse(self.screen, int(screen_pos.x), int(screen_pos.y), 8, 4, self.COLOR_HOLE)
        pygame.gfxdraw.aaellipse(self.screen, int(screen_pos.x), int(screen_pos.y), 8, 4, (50,50,50))
        pole_top = screen_pos - pygame.math.Vector2(0, 40)
        pygame.draw.line(self.screen, self.COLOR_TEXT, (screen_pos.x, screen_pos.y-4), pole_top, 1)
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, [pole_top, pole_top + pygame.math.Vector2(-15, 5), pole_top + pygame.math.Vector2(0, 10)])

    def _render_ball(self, pos):
        screen_pos = self._iso_to_screen(pos.x, pos.y) + self.iso_offset
        shadow_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.gfxdraw.filled_ellipse(shadow_surf, 10, 10, 6, 3, self.COLOR_BALL_SHADOW)
        self.screen.blit(shadow_surf, (int(screen_pos.x) - 10, int(screen_pos.y) - 5))
        pygame.gfxdraw.filled_circle(self.screen, int(screen_pos.x), int(screen_pos.y) - 5, 5, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, int(screen_pos.x), int(screen_pos.y) - 5, 5, (255,255,255))

    def _render_aiming_assists(self):
        if self.power > 0:
            sim_pos = pygame.math.Vector2(self.ball_pos)
            rad_angle = math.radians(self.aim_angle)
            force = self.power / self.MAX_POWER * 15
            sim_vel = pygame.math.Vector2(math.cos(rad_angle) * force, math.sin(rad_angle) * force)
            
            points = []
            for _ in range(30):
                sim_pos += sim_vel
                friction = 0.97
                sim_vel *= friction
                
                if abs(sim_pos.x) > self.WORLD_SIZE: sim_vel.x *= -1
                if abs(sim_pos.y) > self.WORLD_SIZE: sim_vel.y *= -1
                
                if _ % 2 == 0:
                    screen_point = self._iso_to_screen(sim_pos.x, sim_pos.y) + self.iso_offset
                    points.append((int(screen_point.x), int(screen_point.y) - 5))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.COLOR_TRAJECTORY, False, points, 2)
    
    def _render_ui(self):
        hole_text = self.font_small.render(f"Hole: {self.current_hole_num}/{self.NUM_HOLES}", True, self.COLOR_TEXT)
        strokes_text = self.font_small.render(f"Strokes: {self.strokes}/{self.MAX_STROKES}", True, self.COLOR_TEXT)
        self.screen.blit(hole_text, (10, 10))
        self.screen.blit(strokes_text, (self.WIDTH - strokes_text.get_width() - 10, 10))

        if self.game_phase == "aiming":
            bar_width, bar_height = 150, 15
            bar_x, bar_y = (self.WIDTH - bar_width) // 2, self.HEIGHT - 30
            power_ratio = self.power / self.MAX_POWER
            
            pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
            if power_ratio > 0:
                pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_FILL, (bar_x, bar_y, bar_width * power_ratio, bar_height), border_radius=4)
        
        if self.game_over:
            if self.current_hole_num > self.NUM_HOLES:
                msg = "Congratulations!"
                score_msg = f"Total Strokes: {self.total_score}"
            else:
                msg = "Out of Strokes"
                score_msg = f"Hole {self.current_hole_num} failed"
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            msg_render = self.font_large.render(msg, True, self.COLOR_TEXT)
            score_render = self.font_small.render(score_msg, True, self.COLOR_TEXT)
            self.screen.blit(msg_render, (self.WIDTH // 2 - msg_render.get_width() // 2, self.HEIGHT // 2 - 50))
            self.screen.blit(score_render, (self.WIDTH // 2 - score_render.get_width() // 2, self.HEIGHT // 2 + 10))

    def _get_info(self):
        return {
            "score": self.total_score,
            "hole": self.current_hole_num,
            "strokes": self.strokes,
        }

    def _create_spark_effect(self, pos):
        # Sound: "spark.wav"
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = random.randint(10, 25)
            self.particles.append(Particle(pos, vel, (255, 255, 200), lifespan))

    def render(self):
        return self._get_observation()

if __name__ == '__main__':
    # This block allows you to run the game and play it with your keyboard
    # This is not used by the evaluation, but is helpful for testing.
    # To use this, you'll need to run `pip install pygame`
    # and comment out the `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` line.
    
    # The following code is not part of the required solution, but is provided
    # for human-interactive testing.
    
    # To play the game, remove the "dummy" video driver line:
    # del os.environ['SDL_VIDEODRIVER']
    
    env = GameEnv()
    obs, info = env.reset()
    
    # We need to create a display for interactive playing
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Mini-Golf")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        move_action = 0
        if keys[pygame.K_UP]:
            move_action = 1
        elif keys[pygame.K_DOWN]:
            move_action = 2
        elif keys[pygame.K_LEFT]:
            move_action = 3
        elif keys[pygame.K_RIGHT]:
            move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = np.array([move_action, space_action, shift_action])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            # Display the final frame
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            print(f"Game Over. Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds
            
            # Reset the environment
            obs, info = env.reset()

        # The observation is the rendered screen. We need to transpose it for pygame.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Control the frame rate
        if env.game_phase == 'ball_moving':
            clock.tick(60)
        else:
            clock.tick(30)
            
    pygame.quit()