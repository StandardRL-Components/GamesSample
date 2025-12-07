
# Generated: 2025-08-28T03:20:48.591236
# Source Brief: brief_01983.md
# Brief Index: 1983

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: → to run right, ← to run left, ↑ to jump. "
        "Press space when near obstacles for a visual flair."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a speedy robot through a hazardous, procedurally-generated course. "
        "Jump over obstacles, reach the end of 3 stages, and get a high score by performing risky maneuvers."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.STAGE_LENGTH = 6000
        self.TOTAL_STAGES = 3

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_EYE = (255, 255, 255)
        self.COLOR_OBSTACLE_STATIC = (255, 80, 80)
        self.COLOR_OBSTACLE_MOVING = (255, 150, 80)
        self.COLOR_FINISH = (80, 255, 80)
        self.COLOR_PARTICLE_RISK = (255, 255, 100)
        self.COLOR_PARTICLE_JUMP = (180, 180, 180)
        self.COLOR_UI_TEXT = (220, 220, 220)

        # Physics constants
        self.GRAVITY = 0.5
        self.JUMP_STRENGTH = -11
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = 0.92
        self.MAX_VEL_X = 7.0
        self.GROUND_Y = self.SCREEN_HEIGHT - 50

        # Game settings
        self.STAGE_TIME_LIMIT_SECONDS = 60
        self.FPS = 30
        self.STAGE_TIME_LIMIT_STEPS = self.STAGE_TIME_LIMIT_SECONDS * self.FPS
        self.RISK_RADIUS = 60
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
            self.font_big = pygame.font.SysFont("monospace", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 24)
            self.font_big = pygame.font.Font(None, 60)

        # Initialize state variables
        self.player = {}
        self.obstacles = []
        self.particles = []
        self.camera_x = 0.0
        self.finish_x = 0
        self.steps = 0
        self.stage = 1
        self.stage_time_left = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.last_risky_maneuver_step = -100 # Cooldown

        # This will be called once in __init__
        self.reset()
        
        self.validate_implementation()

    def _generate_stage(self):
        self.obstacles.clear()
        self.finish_x = self.STAGE_LENGTH

        # Obstacle speed increases with stage
        base_obstacle_speed = 1.0 + (self.stage - 1) * 0.5
        
        current_x = 800
        while current_x < self.finish_x - 800:
            obstacle_type = self.np_random.integers(0, 3)
            
            if obstacle_type == 0: # Static block
                h = self.np_random.integers(40, 90)
                w = self.np_random.integers(30, 60)
                self.obstacles.append({
                    'rect': pygame.Rect(current_x, self.GROUND_Y - h, w, h),
                    'type': 'static',
                    'color': self.COLOR_OBSTACLE_STATIC
                })
            elif obstacle_type == 1: # Moving platform (vertical)
                h = 20
                w = 100
                travel_dist = self.np_random.integers(80, 150)
                self.obstacles.append({
                    'rect': pygame.Rect(current_x, self.GROUND_Y - h - 50, w, h),
                    'type': 'moving',
                    'color': self.COLOR_OBSTACLE_MOVING,
                    'start_y': self.GROUND_Y - h - 50,
                    'range': travel_dist,
                    'speed': base_obstacle_speed * self.np_random.uniform(0.8, 1.2),
                    'direction': 1
                })
            elif obstacle_type == 2: # Rotating gear
                size = self.np_random.integers(40, 60)
                self.obstacles.append({
                    'pos': pygame.Vector2(current_x, self.GROUND_Y - size - 20),
                    'type': 'rotating',
                    'color': self.COLOR_OBSTACLE_STATIC,
                    'size': size,
                    'angle': self.np_random.uniform(0, math.pi * 2),
                    'rot_speed': (base_obstacle_speed / 25.0) * self.np_random.choice([-1, 1])
                })

            current_x += self.np_random.integers(350, 550)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player = {
            'pos': pygame.Vector2(150, self.GROUND_Y),
            'vel': pygame.Vector2(0, 0),
            'on_ground': True,
            'size': pygame.Vector2(30, 50)
        }
        self.camera_x = 0.0
        self.particles.clear()
        
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.stage_time_left = self.STAGE_TIME_LIMIT_STEPS
        self.game_over = False
        self.game_won = False
        self.last_risky_maneuver_step = -100

        self._generate_stage()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01  # Time penalty
        self.steps += 1
        self.stage_time_left -= 1
        
        # 1. Handle Input
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement)

        # 2. Update Physics
        self._update_physics()

        # 3. Handle Collisions and Game Events
        terminated, event_reward = self._handle_collisions_and_events(space_held)
        reward += event_reward

        # 4. Update Camera
        target_camera_x = self.player['pos'].x - self.SCREEN_WIDTH / 3.5
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

        # 5. Check Termination Conditions
        if self.stage_time_left <= 0:
            terminated = True
        
        if terminated:
            self.game_over = True
            if not self.game_won:
                reward -= 10 # Penalty for failure
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement):
        # movement: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1 and self.player['on_ground']:
            self.player['vel'].y = self.JUMP_STRENGTH
            self.player['on_ground'] = False
            # sfx: jump
            self._create_particles(self.player['pos'] + pygame.Vector2(0, 2), 10, self.COLOR_PARTICLE_JUMP, 2.0)
        
        if movement == 3: # Left
            self.player['vel'].x -= self.PLAYER_ACCEL
        elif movement == 4: # Right
            self.player['vel'].x += self.PLAYER_ACCEL

    def _update_physics(self):
        # Player physics
        if not self.player['on_ground']:
            self.player['vel'].y += self.GRAVITY
        
        self.player['vel'].x *= self.PLAYER_FRICTION
        self.player['vel'].x = np.clip(self.player['vel'].x, -self.MAX_VEL_X, self.MAX_VEL_X)
        self.player['vel'].y = np.clip(self.player['vel'].y, -np.inf, 15.0)

        self.player['pos'] += self.player['vel']
        
        # Ground collision
        if self.player['pos'].y > self.GROUND_Y:
            self.player['pos'].y = self.GROUND_Y
            self.player['vel'].y = 0
            self.player['on_ground'] = True
        
        # World bounds
        if self.player['pos'].x < self.player['size'].x / 2:
            self.player['pos'].x = self.player['size'].x / 2
            self.player['vel'].x = 0

        # Obstacle physics
        for obs in self.obstacles:
            if obs['type'] == 'moving':
                obs['rect'].y += obs['speed'] * obs['direction']
                if obs['rect'].y < obs['start_y'] - obs['range'] or obs['rect'].y > obs['start_y']:
                    obs['direction'] *= -1
            elif obs['type'] == 'rotating':
                obs['angle'] += obs['rot_speed']

        # Particle physics
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1

    def _handle_collisions_and_events(self, space_held):
        terminated = False
        event_reward = 0.0

        player_rect = pygame.Rect(
            self.player['pos'].x - self.player['size'].x / 2,
            self.player['pos'].y - self.player['size'].y,
            self.player['size'].x, self.player['size'].y
        )
        
        # Forward movement reward
        if self.player['vel'].x > 1.0:
            event_reward += 0.1 * (self.player['vel'].x / self.MAX_VEL_X)
        
        # Obstacle interactions
        risky_maneuver_achieved = False
        for obs in self.obstacles:
            obs_collider = None
            if obs['type'] == 'static' or obs['type'] == 'moving':
                obs_collider = obs['rect']
            elif obs['type'] == 'rotating':
                obs_collider = pygame.Rect(obs['pos'].x - obs['size'], obs['pos'].y - obs['size'], obs['size']*2, obs['size']*2)
            
            if obs_collider and player_rect.colliderect(obs_collider):
                # More precise check for rotating obstacle
                if obs['type'] == 'rotating':
                    collided = False
                    for i in range(2):
                        angle = obs['angle'] + i * math.pi / 2
                        p1 = obs['pos'] + pygame.Vector2(obs['size'], 0).rotate_rad(angle)
                        p2 = obs['pos'] - pygame.Vector2(obs['size'], 0).rotate_rad(angle)
                        if player_rect.clipline(p1, p2):
                            collided = True
                            break
                    if not collided: continue

                # sfx: explosion
                terminated = True
                event_reward -= 100.0 # Large penalty for collision
                self._create_particles(self.player['pos'], 50, self.COLOR_PLAYER, 5.0)
                return terminated, event_reward

            # Risky maneuver check
            dist_to_obstacle = player_rect.centerx - obs_collider.centerx
            if abs(dist_to_obstacle) < self.SCREEN_WIDTH / 2: # Only check nearby obstacles
                dist = pygame.Vector2(player_rect.center).distance_to(obs_collider.center)
                if dist < self.RISK_RADIUS + obs_collider.width / 2 and self.steps > self.last_risky_maneuver_step + self.FPS:
                    risky_maneuver_achieved = True
                    self.last_risky_maneuver_step = self.steps
                    event_reward += 1.0
                    if space_held:
                        # sfx: risky_swoosh
                        self._create_particles(player_rect.center, 20, self.COLOR_PARTICLE_RISK, 4.0, 0.5)

        # Stage completion
        if player_rect.centerx > self.finish_x:
            # sfx: stage_complete
            event_reward += 10.0
            self.stage += 1
            if self.stage > self.TOTAL_STAGES:
                self.game_won = True
                terminated = True
                event_reward += 100.0 # Bonus for winning the game
            else:
                # Reset for next stage
                self.player['pos'] = pygame.Vector2(150, self.GROUND_Y)
                self.player['vel'] = pygame.Vector2(0, 0)
                self.stage_time_left = self.STAGE_TIME_LIMIT_STEPS
                self._generate_stage()

        return terminated, event_reward

    def _create_particles(self, pos, count, color, max_speed, lifespan_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifespan': self.np_random.integers(10, 25) * lifespan_mult,
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw background grid
        grid_size = 50
        start_x = int(-self.camera_x % grid_size)
        for x in range(start_x, self.SCREEN_WIDTH, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, grid_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GRID, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

        # Draw particles
        for p in self.particles:
            screen_pos = (int(p['pos'].x - self.camera_x), int(p['pos'].y))
            radius = int(p['lifespan'] / 4)
            if radius > 1:
                pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], radius, p['color'])

        # Draw finish line
        finish_screen_x = self.finish_x - self.camera_x
        if 0 < finish_screen_x < self.SCREEN_WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_screen_x, 0), (finish_screen_x, self.SCREEN_HEIGHT), 5)
        
        # Draw obstacles
        for obs in self.obstacles:
            if obs['type'] == 'static' or obs['type'] == 'moving':
                screen_rect = obs['rect'].move(-self.camera_x, 0)
                pygame.draw.rect(self.screen, obs['color'], screen_rect, border_radius=3)
            elif obs['type'] == 'rotating':
                screen_pos = (obs['pos'].x - self.camera_x, obs['pos'].y)
                if -100 < screen_pos[0] < self.SCREEN_WIDTH + 100:
                    for i in range(2):
                        angle = obs['angle'] + i * math.pi / 2
                        p1 = pygame.Vector2(screen_pos) + pygame.Vector2(obs['size'], 0).rotate_rad(angle)
                        p2 = pygame.Vector2(screen_pos) - pygame.Vector2(obs['size'], 0).rotate_rad(angle)
                        pygame.draw.line(self.screen, obs['color'], p1, p2, 6)
        
        # Draw player
        if not self.game_over or self.game_won:
            p_rect = pygame.Rect(
                self.player['pos'].x - self.player['size'].x / 2 - self.camera_x,
                self.player['pos'].y - self.player['size'].y,
                self.player['size'].x, self.player['size'].y
            )
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, p_rect, border_radius=8)
            
            # Eye to show direction
            eye_dir = 1 if self.player['vel'].x >= 0 else -1
            eye_pos = (
                int(p_rect.centerx + eye_dir * 5),
                int(p_rect.centery - 10)
            )
            pygame.draw.circle(self.screen, self.COLOR_PLAYER_EYE, eye_pos, 4)

    def _render_ui(self):
        # Stage
        stage_text = self.font_ui.render(f"STAGE: {self.stage}/{self.TOTAL_STAGES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Time
        time_text = self.font_ui.render(f"TIME: {self.stage_time_left // self.FPS:02d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score):,}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))

        # Game Over / Win Text
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_FINISH if self.game_won else self.COLOR_OBSTACLE_STATIC
            end_text = self.font_big.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - end_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_left": self.stage_time_left / self.FPS
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("RoboRun")
    clock = pygame.time.Clock()

    print(env.user_guide)
    print(env.game_description)

    while not terminated:
        # Human controls
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

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render to screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        if info.get('score', 0) != env.score:
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}")

        clock.tick(env.FPS)

    print(f"Final Score: {info['score']:.2f} in {info['steps']} steps.")
    pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()