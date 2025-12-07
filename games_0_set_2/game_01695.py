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
        "Controls: Arrows to move the cursor. Press Space to draw a line from the rider to the cursor. Hold Shift to reduce bounce on collisions."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw lines to guide a physics-based rider to the finish line in a procedurally generated landscape. Reach the goal before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT = 15.0  # seconds

        # Physics constants
        self.GRAVITY = 0.4
        self.RIDER_RADIUS = 8
        self.CURSOR_SPEED = 15
        self.FRICTION = 0.99
        self.BOUNCE = 0.7
        self.DRIFT_BOUNCE = 0.2
        
        # Gameplay constants
        self.MAX_PLAYER_LINES = 5
        self.MAX_STEPS = int(self.TIME_LIMIT * self.FPS * 1.5) # Generous step limit

        # Colors
        self.COLOR_BG = (15, 18, 22)
        self.COLOR_GRID = (30, 35, 45)
        self.COLOR_TERRAIN = (100, 110, 120)
        self.COLOR_PLAYER_LINE = (0, 200, 255)
        self.COLOR_RIDER = (255, 255, 255)
        self.COLOR_RIDER_GLOW = (0, 200, 255)
        self.COLOR_CURSOR = (255, 0, 100)
        self.COLOR_START = (0, 255, 120)
        self.COLOR_FINISH = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_PARTICLE = (255, 255, 255)

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 36, bold=True)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_time = 0.0
        self.rider_pos = None
        self.rider_vel = None
        self.rider_trail = []
        self.cursor_pos = None
        self.terrain_lines = []
        self.player_lines = []
        self.particles = []
        self.finish_x = 0
        self.last_dist_to_finish = 0
        self.prev_space_held = False
        self.win_message = ""

        # Use a seeded random number generator
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_time = 0.0
        self.win_message = ""

        # Procedurally generate terrain
        self.terrain_lines = []
        start_y = self.np_random.uniform(150, 250)
        points = [(0, start_y)]
        num_segments = 5
        for i in range(1, num_segments):
            x = i * (self.WIDTH / num_segments)
            y = self.np_random.uniform(100, self.HEIGHT - 50)
            points.append((x, y))
        points.append((self.WIDTH, self.np_random.uniform(150, 250)))

        for i in range(len(points) - 1):
            self.terrain_lines.append(
                (np.array(points[i], dtype=float), np.array(points[i+1], dtype=float))
            )

        # --- FIX: Place rider on the initial terrain to pass stability test ---
        rider_start_x = 50.0
        # The rider starts on the first segment of the terrain.
        first_segment_p1, first_segment_p2 = self.terrain_lines[0]
        # Interpolate the y-coordinate of the terrain at the rider's starting x.
        # This prevents the rider from spawning in mid-air and immediately falling out of bounds.
        if (first_segment_p2[0] - first_segment_p1[0]) != 0:
            t = (rider_start_x - first_segment_p1[0]) / (first_segment_p2[0] - first_segment_p1[0])
            rider_start_y = first_segment_p1[1] + t * (first_segment_p2[1] - first_segment_p1[1])
        else: # Fallback for vertical line, though not expected from generation
            rider_start_y = first_segment_p1[1]
        
        # Reset rider, placing it just above the calculated terrain point
        self.rider_pos = np.array([rider_start_x, rider_start_y - self.RIDER_RADIUS - 1.0])
        self.rider_vel = np.array([self.np_random.uniform(3, 6), 0.0])
        self.rider_trail = []
        # --- END FIX ---

        # Reset gameplay elements
        self.cursor_pos = self.rider_pos + np.array([100.0, 0.0])
        self.player_lines = []
        self.particles = []
        self.finish_x = self.WIDTH - 50
        self.last_dist_to_finish = abs(self.rider_pos[0] - self.finish_x)
        self.prev_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, True, self._get_info()
            
        self.clock.tick(self.FPS)
        self.game_time += 1 / self.FPS
        self.steps += 1

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle player input ---
        self._update_cursor(movement)
        self._handle_actions(space_held)

        # --- Update game logic ---
        self._update_rider_physics(shift_held)
        self._update_particles()
        
        # --- Calculate reward and check termination ---
        reward, terminated, truncated = self._calculate_reward_and_termination()
        
        if terminated or truncated:
            self.game_over = True
            if self.rider_pos[0] >= self.finish_x:
                self.win_message = "FINISH!"
            elif terminated:
                self.win_message = "TRY AGAIN"
            else: # Truncated
                self.win_message = "TIME UP"

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _update_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)
        
    def _handle_actions(self, space_held):
        is_space_press = space_held and not self.prev_space_held
        if is_space_press:
            # sfx: draw_line.wav
            start_point = self.rider_pos.copy()
            end_point = self.cursor_pos.copy()
            self.player_lines.append((start_point, end_point))
            if len(self.player_lines) > self.MAX_PLAYER_LINES:
                self.player_lines.pop(0)
        self.prev_space_held = space_held

    def _update_rider_physics(self, shift_held):
        # Apply gravity and friction
        self.rider_vel[1] += self.GRAVITY
        self.rider_vel *= self.FRICTION

        # Update position
        self.rider_pos += self.rider_vel

        # Update trail
        self.rider_trail.append(self.rider_pos.copy())
        if len(self.rider_trail) > 20:
            self.rider_trail.pop(0)

        # Collision detection
        all_lines = self.terrain_lines + self.player_lines
        for p1, p2 in all_lines:
            # Line-circle collision check
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0: continue

            t = np.dot(self.rider_pos - p1, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)
            
            closest_point = p1 + t * line_vec
            dist_vec = self.rider_pos - closest_point
            dist_sq = np.dot(dist_vec, dist_vec)

            if dist_sq < self.RIDER_RADIUS ** 2:
                # sfx: impact_light.wav or scrape.wav
                self._create_particles(self.rider_pos, 5)
                
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 1
                normal = dist_vec / dist
                
                # Resolve penetration
                self.rider_pos = closest_point + normal * self.RIDER_RADIUS
                
                # Reflect velocity
                restitution = self.DRIFT_BOUNCE if shift_held else self.BOUNCE
                vel_comp = np.dot(self.rider_vel, normal)
                self.rider_vel -= (1 + restitution) * vel_comp * normal

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[2] > 0]
        for p in self.particles:
            p[0] += p[1]  # Update position
            p[2] -= 1  # Decrease lifespan

    def _calculate_reward_and_termination(self):
        terminated = False
        truncated = False
        reward = 0

        # Proximity reward
        current_dist = abs(self.rider_pos[0] - self.finish_x)
        reward += (self.last_dist_to_finish - current_dist) * 0.1 # Reward for getting closer
        self.last_dist_to_finish = current_dist

        # Win condition (termination)
        if self.rider_pos[0] >= self.finish_x:
            reward += 50.0  # Large reward for winning
            self.score += 1
            terminated = True
            # sfx: win_ fanfare.wav

        # Loss condition (termination)
        if not (0 < self.rider_pos[0] < self.WIDTH and -self.RIDER_RADIUS < self.rider_pos[1] < self.HEIGHT + self.RIDER_RADIUS):
            reward -= 5.0  # Penalty for crashing
            terminated = True
            # sfx: fail_sound.wav
        
        # Time/step limits (truncation)
        if self.game_time >= self.TIME_LIMIT:
            if not terminated: # Don't add penalty if already won/crashed
                reward -= 5.0 # Penalty for timeout
            truncated = True
            # sfx: timeout_buzzer.wav

        if self.steps >= self.MAX_STEPS:
            truncated = True
            
        return reward, terminated, truncated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Start and Finish lines
        pygame.draw.line(self.screen, self.COLOR_START, (50, 0), (50, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_x, 0), (self.finish_x, self.HEIGHT), 2)

        # Lines
        for p1, p2 in self.terrain_lines:
            pygame.draw.aaline(self.screen, self.COLOR_TERRAIN, p1, p2, 3)
        for p1, p2 in self.player_lines:
            pygame.draw.aaline(self.screen, self.COLOR_PLAYER_LINE, p1, p2, 2)
            
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(p[2] * 15)))
            color = (*self.COLOR_PARTICLE, alpha)
            s = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect())
            self.screen.blit(s, p[0])

        # Rider Trail
        if self.rider_trail:
            for i, pos in enumerate(self.rider_trail):
                alpha = int(200 * (i / len(self.rider_trail)))
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.RIDER_RADIUS, (*self.COLOR_RIDER_GLOW, alpha))

        # Rider
        rider_x, rider_y = int(self.rider_pos[0]), int(self.rider_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, self.RIDER_RADIUS + 3, (*self.COLOR_RIDER_GLOW, 100))
        pygame.gfxdraw.filled_circle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)
        pygame.gfxdraw.aacircle(self.screen, rider_x, rider_y, self.RIDER_RADIUS, self.COLOR_RIDER)

        # Cursor
        if not self.game_over:
            cursor_x, cursor_y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_x - 8, cursor_y), (cursor_x + 8, cursor_y), 2)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y - 8), (cursor_x, cursor_y + 8), 2)
            
            # Line preview
            pygame.draw.aaline(self.screen, self.COLOR_CURSOR, self.rider_pos, self.cursor_pos)


    def _render_ui(self):
        # Time remaining
        time_left = max(0, self.TIME_LIMIT - self.game_time)
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Speed
        speed = np.linalg.norm(self.rider_vel)
        speed_text = self.font_ui.render(f"SPEED: {speed:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.WIDTH - speed_text.get_width() - 10, 10))
        
        # Game over message
        if self.game_over and self.win_message:
            msg_surf = self.font_msg.render(self.win_message, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time": self.game_time,
            "rider_pos": self.rider_pos.tolist(),
            "rider_vel": self.rider_vel.tolist(),
        }

    def _create_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            lifespan = self.np_random.integers(10, 25)
            self.particles.append([pos.copy(), vel, lifespan])

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Re-enable video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()

    obs, info = env.reset(seed=42)
    
    # Set up window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Line Rider Gym")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset(seed=42)
                total_reward = 0
                done = False

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    print(f"Episode finished. Total reward: {total_reward}")
    env.close()