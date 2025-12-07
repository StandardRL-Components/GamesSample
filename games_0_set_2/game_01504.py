import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    user_guide = (
        "Controls: ↑ to accelerate, ↓ to decelerate, ←→ to turn. "
        "Hold SPACE for a speed boost. Hold SHIFT to brake."
    )

    game_description = (
        "A fast-paced, top-down arcade racer. Navigate a procedurally generated "
        "track, avoid obstacles, and reach the finish line of three challenging stages "
        "before time runs out."
    )

    auto_advance = True

    # --- Constants ---
    # Colors (Neon Palette)
    COLOR_BG = (15, 20, 30)
    COLOR_TRACK = (40, 50, 70)
    COLOR_LINES = (200, 200, 220)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255)
    COLOR_OBSTACLE = (255, 0, 100)
    COLOR_OBSTACLE_GLOW = (200, 0, 80)
    COLOR_BOOST = (255, 255, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_FINISH_1 = (255, 255, 255)
    COLOR_FINISH_2 = (100, 100, 100)

    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game Parameters
    TRACK_WIDTH = 500
    STAGE_LENGTH = 5000
    NUM_STAGES = 3
    TIME_PER_STAGE = 60  # seconds
    MAX_EPISODE_STEPS = (TIME_PER_STAGE * NUM_STAGES) * metadata["render_fps"]

    # Player Physics
    PLAYER_SIZE = 12
    ACCELERATION = 0.25
    FRICTION = 0.97
    TURN_SPEED = 0.07
    MAX_SPEED = 6.0
    BRAKE_FORCE = 0.5
    BOOST_SPEED = 12.0
    BOOST_ACCEL = 0.8

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24)

        self.render_mode = render_mode
        self.np_random = None

        self.player_pos = None
        self.player_speed = None
        self.player_angle = None
        self.obstacles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.stage = None
        self.time_left = None
        self.game_over = None
        self.win = None
        self.last_reward_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([0.0, 0.0])
        self.player_speed = 0.0
        self.player_angle = 0.0

        self.steps = 0
        self.score = 0
        self.stage = 1
        self.time_left = self.TIME_PER_STAGE * self.NUM_STAGES * self.metadata["render_fps"]
        self.game_over = False
        self.win = False

        self.obstacles = []
        self.particles = []
        self._generate_stage_obstacles()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        reward = self._update_game_state(movement, space_held, shift_held)
        
        self.score += reward
        self.steps += 1
        self.time_left -= 1

        terminated = self._check_termination()
        truncated = False # This environment does not truncate
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

    def _update_game_state(self, movement, space_held, shift_held):
        # --- Update Player ---
        is_turning = movement in [3, 4]
        if is_turning:
            turn_direction = -1 if movement == 3 else 1
            self.player_angle += turn_direction * self.TURN_SPEED

        accel_input = 0
        if movement == 1:
            accel_input = 1
        elif movement == 2:
            accel_input = -0.5

        if space_held: # Boost
            self.player_speed = min(self.BOOST_SPEED, self.player_speed + self.BOOST_ACCEL)
            self._create_particles(self.player_pos, 5, self.COLOR_BOOST, 2, 8)
        elif shift_held: # Brake
            self.player_speed = max(0, self.player_speed - self.BRAKE_FORCE)
        else: # Normal acceleration
            self.player_speed += accel_input * self.ACCELERATION
            self.player_speed = min(self.MAX_SPEED, self.player_speed)

        self.player_speed *= self.FRICTION
        if abs(self.player_speed) < 0.01:
            self.player_speed = 0

        # Update position
        velocity = np.array([
            math.sin(self.player_angle) * self.player_speed,
            -math.cos(self.player_angle) * self.player_speed
        ])
        self.player_pos += velocity

        # Clamp to track boundaries
        half_track = self.TRACK_WIDTH / 2
        self.player_pos[0] = np.clip(self.player_pos[0], -half_track, half_track)

        # --- Update Particles ---
        self._update_particles()
        if self.player_speed > 1:
            self._create_particles(self.player_pos, 2, self.COLOR_PLAYER, 1, self.player_speed / 2)

        # --- Check Collisions and Progress ---
        collided = self._check_collisions()
        stage_cleared = self._check_stage_clear()

        # --- Calculate Reward ---
        reward = 0.1  # Survival reward

        # Penalty for being "safe" (slow and not turning)
        is_safe_action = self.player_speed < self.MAX_SPEED * 0.3 and not is_turning
        if is_safe_action:
            reward -= 2.0

        if collided:
            reward = -100.0
            self.game_over = True
        
        if stage_cleared:
            if self.stage == self.NUM_STAGES: # Final stage cleared
                reward += 100.0
                self.win = True
                self.game_over = True
            else: # Intermediate stage cleared
                reward += 10.0
                self.stage += 1
                self._generate_stage_obstacles()
        
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.time_left <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            return True
        return False

    def _generate_stage_obstacles(self):
        self.obstacles.clear()
        obstacle_density = 0.0005 + (self.stage - 1) * 0.00005 # Increase density by 10% per stage
        
        stage_start_y = - (self.stage - 1) * self.STAGE_LENGTH
        stage_end_y = - self.stage * self.STAGE_LENGTH

        num_obstacles = int(self.STAGE_LENGTH * self.TRACK_WIDTH * obstacle_density)
        
        for _ in range(num_obstacles):
            x = self.np_random.uniform(-self.TRACK_WIDTH / 2, self.TRACK_WIDTH / 2)
            # FIX: Swapped arguments to np.random.uniform to ensure low < high.
            # The y-axis is inverted, so stage_end_y is a smaller number than stage_start_y.
            low_y = stage_end_y + 200
            high_y = stage_start_y - 200
            y = self.np_random.uniform(low_y, high_y)
            self.obstacles.append(np.array([x, y]))

    def _check_collisions(self):
        player_rect = pygame.Rect(
            self.SCREEN_WIDTH / 2 - self.PLAYER_SIZE,
            self.SCREEN_HEIGHT / 2 - self.PLAYER_SIZE,
            self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2
        )
        for obs_pos in self.obstacles:
            obs_screen_pos = obs_pos - self.player_pos
            if abs(obs_screen_pos[0]) > self.SCREEN_WIDTH / 2 + 20 or abs(obs_screen_pos[1]) > self.SCREEN_HEIGHT / 2 + 20:
                continue

            obs_rect = pygame.Rect(
                self.SCREEN_WIDTH / 2 + obs_screen_pos[0] - 10,
                self.SCREEN_HEIGHT / 2 + obs_screen_pos[1] - 10,
                20, 20
            )
            if player_rect.colliderect(obs_rect):
                return True
        return False

    def _check_stage_clear(self):
        stage_finish_y = -self.stage * self.STAGE_LENGTH
        return self.player_pos[1] < stage_finish_y

    def _create_particles(self, pos, count, color, size_base, speed_factor):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_factor
            velocity = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifetime = self.np_random.integers(10, 20)
            self.particles.append({
                'pos': pos.copy(),
                'vel': velocity,
                'life': lifetime,
                'max_life': lifetime,
                'color': color,
                'size': self.np_random.uniform(size_base, size_base * 2)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "time_left": int(self.time_left / self.metadata["render_fps"]),
            "player_speed": self.player_speed,
        }

    def _render_game(self):
        # --- Camera setup ---
        camera_offset = self.player_pos - np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2])

        # --- Render Track ---
        track_rect = pygame.Rect(
            -camera_offset[0],
            -camera_offset[1] - self.player_pos[1],
            self.TRACK_WIDTH,
            self.SCREEN_HEIGHT * 10 # A very long rect to cover scrolling
        )
        track_rect.centerx = self.SCREEN_WIDTH / 2
        pygame.draw.rect(self.screen, self.COLOR_TRACK, track_rect)

        # Center line
        line_length = 40
        gap_length = 20
        total_segment_length = line_length + gap_length
        scroll_offset = self.player_pos[1] % total_segment_length
        
        for y in range(-line_length, self.SCREEN_HEIGHT + line_length, total_segment_length):
            start_y = y - scroll_offset
            end_y = start_y + line_length
            pygame.draw.line(self.screen, self.COLOR_LINES, (self.SCREEN_WIDTH/2, start_y), (self.SCREEN_WIDTH/2, end_y), 3)

        # --- Render Finish Lines ---
        for i in range(1, self.NUM_STAGES + 1):
            finish_y = -i * self.STAGE_LENGTH
            screen_y = self.SCREEN_HEIGHT / 2 + (finish_y - self.player_pos[1])
            if -20 < screen_y < self.SCREEN_HEIGHT + 20:
                self._render_chequered_line(screen_y)

        # --- Render Obstacles ---
        for obs_pos in self.obstacles:
            screen_pos = self.SCREEN_WIDTH/2 + (obs_pos - self.player_pos)
            if -20 < screen_pos[0] < self.SCREEN_WIDTH + 20 and -20 < screen_pos[1] < self.SCREEN_HEIGHT + 20:
                rect = pygame.Rect(0, 0, 20, 20)
                rect.center = (int(screen_pos[0]), int(screen_pos[1]))
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect, border_radius=3)
                pygame.gfxdraw.rectangle(self.screen, rect, self.COLOR_OBSTACLE_GLOW)


        # --- Render Particles ---
        for p in self.particles:
            screen_pos = self.SCREEN_WIDTH/2 + (p['pos'] - self.player_pos)
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                try:
                    # Use a surface to handle alpha blending correctly
                    s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.gfxdraw.filled_circle(s, size, size, size, color)
                    pygame.gfxdraw.aacircle(s, size, size, size, color)
                    self.screen.blit(s, (int(screen_pos[0] - size), int(screen_pos[1] - size)))
                except (ValueError, TypeError):
                    # In some edge cases size/color can be invalid, just skip the particle
                    pass


        # --- Render Player ---
        player_screen_pos = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        # Glow
        glow_size = int(self.PLAYER_SIZE * (1.5 + 0.5 * math.sin(self.steps * 0.2)))
        glow_alpha = 100
        try:
            # Use a surface to handle alpha blending correctly
            s = pygame.Surface((glow_size*2, glow_size*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, glow_size, glow_size, glow_size, (*self.COLOR_PLAYER_GLOW, glow_alpha))
            self.screen.blit(s, (int(player_screen_pos[0] - glow_size), int(player_screen_pos[1] - glow_size)))
        except (ValueError, TypeError):
             pass

        # Triangle body
        points = []
        for i in range(3):
            angle = self.player_angle + i * (2 * math.pi / 3)
            x = player_screen_pos[0] + self.PLAYER_SIZE * math.sin(angle)
            y = player_screen_pos[1] - self.PLAYER_SIZE * math.cos(angle)
            points.append((x, y))
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_chequered_line(self, y):
        track_left_x = self.SCREEN_WIDTH/2 - self.TRACK_WIDTH/2
        check_size = 20
        for i, x in enumerate(range(int(track_left_x), int(track_left_x + self.TRACK_WIDTH), check_size)):
            color = self.COLOR_FINISH_1 if i % 2 == 0 else self.COLOR_FINISH_2
            pygame.draw.rect(self.screen, color, (x, y, check_size, 10))

    def _render_ui(self):
        # Stage Text
        stage_text = self.font_small.render(f"STAGE: {self.stage}/{self.NUM_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Timer Text
        time_str = f"TIME: {max(0, int(self.time_left / self.metadata['render_fps']))}"
        timer_text = self.font_small.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Score Text
        score_str = f"SCORE: {int(self.score)}"
        score_text = self.font_medium.render(score_str, True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH/2 - score_text.get_width()/2, self.SCREEN_HEIGHT - 40))

        # Game Over/Win Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(end_text, (self.SCREEN_WIDTH/2 - end_text.get_width()/2, self.SCREEN_HEIGHT/2 - end_text.get_height()/2))


    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    # Make sure to remove the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Arcade Racer")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press R to reset
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata["render_fps"])

    env.close()