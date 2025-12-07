import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to run, ↑ or Space to jump. Reach the white finish line!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling platformer. Guide the robot through a "
        "procedurally generated obstacle course to the finish line as fast as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.LEVEL_WIDTH = 4000
        self.GROUND_Y = 350
        self.FPS = 30
        self.MAX_STEPS = 1000  # As per brief

        # Physics constants
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15
        self.PLAYER_ACCEL = 1.2
        self.PLAYER_FRICTION = -0.1
        self.PLAYER_MAX_SPEED = 8

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_GROUND = (60, 40, 40)
        self.COLOR_PLAYER = (60, 180, 255)
        self.COLOR_PLAYER_GLOW = (120, 210, 255)
        self.COLOR_FINISH = (255, 255, 255)
        self.OBSTACLE_COLORS = {
            'bar': (255, 80, 80),
            'platform': (80, 255, 80),
            'projectile': (255, 255, 80),
        }

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.camera_x = None
        self.obstacles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.difficulty_multiplier = None
        self.last_dist_to_finish = None
        self.jumped_obstacles = None
        self.last_player_x = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([100.0, self.GROUND_Y - 40.0])
        self.player_vel = np.array([0.0, 0.0])
        self.on_ground = True
        self.last_player_x = self.player_pos[0]

        self.camera_x = 0
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.difficulty_multiplier = 1.0

        self._generate_level()

        self.last_dist_to_finish = self.LEVEL_WIDTH - self.player_pos[0]
        self.jumped_obstacles = set()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.obstacles = []
        current_x = 500
        obstacle_id = 0
        while current_x < self.LEVEL_WIDTH - 500:
            current_x += self.np_random.integers(250, 450)
            obstacle_type = self.np_random.choice(['bar', 'platform', 'projectile'])

            if obstacle_type == 'bar':
                height = self.np_random.integers(50, 150)
                self.obstacles.append({
                    'id': obstacle_id, 'type': 'bar',
                    'rect': pygame.Rect(current_x, self.GROUND_Y - height, 20, height),
                    'color': self.OBSTACLE_COLORS['bar']
                })
            elif obstacle_type == 'platform':
                width = self.np_random.integers(80, 150)
                self.obstacles.append({
                    'id': obstacle_id, 'type': 'platform',
                    'rect': pygame.Rect(current_x, self.GROUND_Y - self.np_random.integers(80, 150), width, 20),
                    'color': self.OBSTACLE_COLORS['platform'],
                    'offset': self.np_random.uniform(0, 2 * math.pi),
                    'range': self.np_random.integers(30, 60),
                    'speed': self.np_random.uniform(0.05, 0.1)
                })
            elif obstacle_type == 'projectile':
                size = self.np_random.integers(20, 40)
                self.obstacles.append({
                    'id': obstacle_id, 'type': 'projectile',
                    'rect': pygame.Rect(current_x, self.GROUND_Y - self.np_random.integers(50, 200), size, size),
                    'color': self.OBSTACLE_COLORS['projectile'],
                    'speed': self.np_random.choice([-2, 2]),
                    'initial_x': current_x
                })
            obstacle_id += 1

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1

        self._handle_input(movement, space_held)
        self._update_player()
        self._update_obstacles()
        self._update_particles()

        self._update_camera()
        self._check_collisions()
        self._check_win_condition()

        self.steps += 1
        self.difficulty_multiplier = 1.0 + 0.05 * (self.steps // 500)

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        reward = self._calculate_reward(terminated)
        self.score += reward

        self.last_player_x = self.player_pos[0]
        
        truncated = False # Not used in this game

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] -= self.PLAYER_ACCEL
        if movement == 4:  # Right
            self.player_vel[0] += self.PLAYER_ACCEL

        # Jumping
        if (movement == 1 or space_held) and self.on_ground:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump
            # Spawn jump particles
            for _ in range(10):
                self.particles.append({
                    'pos': self.player_pos + np.array([15, 38]),
                    'vel': np.array([self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 0)]),
                    'lifespan': 15,
                    'color': self.COLOR_GROUND,
                    'size': self.np_random.integers(2, 5)
                })

    def _update_player(self):
        # Apply friction
        if self.player_vel[0] > 0:
            self.player_vel[0] += self.PLAYER_FRICTION * self.player_vel[0]
            if self.player_vel[0] < 0: self.player_vel[0] = 0
        elif self.player_vel[0] < 0:
            self.player_vel[0] += self.PLAYER_FRICTION * self.player_vel[0]
            if self.player_vel[0] > 0: self.player_vel[0] = 0

        # Clamp speed
        self.player_vel[0] = np.clip(self.player_vel[0], -self.PLAYER_MAX_SPEED, self.PLAYER_MAX_SPEED)

        # Apply gravity
        if not self.on_ground:
            self.player_vel[1] += self.GRAVITY

        # Update position
        self.player_pos += self.player_vel

        # Ground collision
        if self.player_pos[1] > self.GROUND_Y - 40:
            if not self.on_ground:
                # sfx: land
                self.on_ground = True
            self.player_pos[1] = self.GROUND_Y - 40
            self.player_vel[1] = 0

        # World bounds
        self.player_pos[0] = max(0, self.player_pos[0])

        # Running particles
        if self.on_ground and abs(self.player_vel[0]) > 2 and self.steps % 3 == 0:
            self.particles.append({
                'pos': self.player_pos + np.array([5, 38]),
                'vel': np.array([-self.player_vel[0] * 0.5, self.np_random.uniform(-0.5, 0)]),
                'lifespan': 10,
                'color': (80, 60, 60),
                'size': self.np_random.integers(1, 4)
            })

    def _update_obstacles(self):
        for o in self.obstacles:
            if o['type'] == 'platform':
                o['rect'].y = o['rect'].y + math.sin(self.steps * o['speed'] * self.difficulty_multiplier + o['offset']) * o['range'] * 0.1
            elif o['type'] == 'projectile':
                o['rect'].x += o['speed'] * self.difficulty_multiplier
                if abs(o['rect'].x - o['initial_x']) > 300:
                    o['rect'].x = o['initial_x']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] - 0.2)

    def _update_camera(self):
        target_camera_x = self.player_pos[0] - self.WIDTH / 2 + 100
        # Smooth camera movement
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        self.camera_x = max(0, min(self.camera_x, self.LEVEL_WIDTH - self.WIDTH))

    def _check_collisions(self):
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], 30, 40)
        for o in self.obstacles:
            if player_rect.colliderect(o['rect']):
                self.game_over = True
                # sfx: explosion
                # Spawn explosion particles
                for _ in range(50):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 8)
                    self.particles.append({
                        'pos': self.player_pos + np.array([15, 20]),
                        'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                        'lifespan': self.np_random.integers(20, 40),
                        'color': self.OBSTACLE_COLORS[self.np_random.choice(list(self.OBSTACLE_COLORS.keys()))],
                        'size': self.np_random.integers(2, 6)
                    })
                return

    def _check_win_condition(self):
        if self.player_pos[0] >= self.LEVEL_WIDTH - 200:
            self.game_over = True
            self.win = True
            # sfx: win_jingle

    def _calculate_reward(self, terminated):
        reward = 0.0

        if terminated:
            if self.win:
                return 100.0
            else:  # Collision or timeout
                return -10.0

        # Reward for moving towards the finish line
        current_dist = self.LEVEL_WIDTH - self.player_pos[0]
        reward += (self.last_dist_to_finish - current_dist) * 0.1
        self.last_dist_to_finish = current_dist

        # Time penalty
        reward -= 0.02

        # Reward for jumping over obstacles
        for o in self.obstacles:
            if o['id'] not in self.jumped_obstacles:
                obs_center_x = o['rect'].centerx
                if self.last_player_x < obs_center_x and self.player_pos[0] >= obs_center_x and not self.on_ground:
                    reward += 1.0
                    self.jumped_obstacles.add(o['id'])
                    # sfx: point_pickup

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Draw ground
        ground_rect = pygame.Rect(0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)

        # Draw finish line
        finish_x = self.LEVEL_WIDTH - 200 - self.camera_x
        if finish_x < self.WIDTH:
            for i in range(10):
                color = self.COLOR_FINISH if i % 2 == 0 else (100, 100, 100)
                pygame.draw.rect(self.screen, color, (finish_x, self.GROUND_Y - i * 20, 20, 20))

        # Draw obstacles
        for o in self.obstacles:
            screen_rect = o['rect'].move(-self.camera_x, 0)
            if screen_rect.right > 0 and screen_rect.left < self.WIDTH:
                pygame.draw.rect(self.screen, o['color'], screen_rect)
                pygame.draw.rect(self.screen, tuple(c * 0.7 for c in o['color']), screen_rect, 2)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            pygame.draw.circle(self.screen, p['color'], pos, int(p['size']))

        # Draw player
        if not (self.game_over and not self.win):
            player_screen_x = int(self.player_pos[0] - self.camera_x)
            player_screen_y = int(self.player_pos[1])
            player_rect = pygame.Rect(player_screen_x, player_screen_y, 30, 40)

            # Glow effect
            glow_surf = pygame.Surface((50, 60), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (25, 30), 20)
            pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 30), (25, 30), 25)
            self.screen.blit(glow_surf, (player_screen_x - 10, player_screen_y - 10))

            # Main body
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
            # Eye
            eye_x = player_screen_x + (20 if self.player_vel[0] >= 0 else 5)
            pygame.draw.rect(self.screen, (255, 255, 255), (eye_x, player_screen_y + 10, 5, 5))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, (255, 255, 255))
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "win": self.win,
            "player_x": self.player_pos[0],
            "distance_to_goal": max(0, self.LEVEL_WIDTH - 200 - self.player_pos[0])
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will create a window and render the game
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robo-Runner")

    terminated = False
    total_reward = 0

    # Game loop
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Keyboard controls for human play
        keys = pygame.key.get_pressed()
        movement = 0  # none
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        if terminated:
            # If the game is over, wait for a key press to reset
            if any(keys):
                obs, info = env.reset()
                terminated = False
                total_reward = 0
        else:
            # Advance the game state
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Control the frame rate
        env.clock.tick(env.FPS)

    env.close()