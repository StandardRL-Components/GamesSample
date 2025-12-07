import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An expert-crafted Gymnasium environment based on the 'Recursive Line Split' design brief.

    **Gameplay:**
    - The player controls the launch angle of a neon green line.
    - Pressing 'Space' (action[1]) launches the line at the current angle and speed.
    - If the line hits a grey obstacle, it splits into two, the score increases,
      the base speed for the next launch increases, and a new obstacle is added.
    - If the launched line(s) fail to hit any obstacles, the base speed decreases.
    - The goal is to reach 20 splits. The game ends if the speed drops too low.

    **Action Space `MultiDiscrete([5, 2, 2])`:**
    - `action[0]` (Movement): 0=None, 1=Up (fine tune angle-), 2=Down (fine tune angle+),
                         3=Left (coarse tune angle-), 4=Right (coarse tune angle+).
    - `action[1]` (Space): 0=Released, 1=Pressed. Launches the line on a press.
    - `action[2]` (Shift): 0=Released, 1=Pressed. No effect.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Launch a line to hit obstacles, causing it to split into more lines. "
        "The goal is to reach 20 splits before your launch speed drops too low."
    )
    user_guide = (
        "Controls: Use ←→ (coarse) and ↑↓ (fine) arrow keys to aim. "
        "Press space to launch the line."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2000
    WIN_SPLIT_COUNT = 20
    INITIAL_OBSTACLES = 3

    # --- Colors ---
    COLOR_BG = (10, 15, 25)
    COLOR_LINE = (50, 255, 150)
    COLOR_OBSTACLE = (80, 90, 110)
    COLOR_PARTICLE = (255, 255, 100)
    COLOR_UI = (220, 220, 240)
    COLOR_AIM = (255, 255, 255, 100) # RGBA

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
        try:
            self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
            self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 32)
        
        self.render_mode = render_mode

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.split_count = 0
        self.game_phase = 'aiming'
        self.lines = []
        self.obstacles = []
        self.particles = []
        self.base_speed = 0.0
        self.launch_angle = 0.0
        self.split_this_turn = False
        self.prev_space_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.split_count = 0
        self.game_phase = 'aiming'
        self.lines = []
        self.obstacles = []
        self.particles = []
        self.base_speed = 8.0
        self.launch_angle = -90.0  # Straight up
        self.split_this_turn = False
        self.prev_space_held = False

        self._spawn_obstacles(self.INITIAL_OBSTACLES)

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0
        terminated = False
        truncated = False

        if self.game_phase == 'aiming':
            self._handle_aiming_input(movement)
            
            # Launch on space press (transition from not held to held)
            if space_held and not self.prev_space_held:
                self._launch_line()
                self.game_phase = 'moving'
                self.split_this_turn = False

        elif self.game_phase == 'moving':
            reward += 0.01  # Small reward for being in motion
            self._update_lines()
            self._update_particles()
            
            # Check if turn is over (no more active lines)
            if not self.lines:
                if not self.split_this_turn:
                    self.base_speed *= 0.95 # Speed penalty for missing
                self.game_phase = 'aiming'
                self.particles.clear() # Clear particles for next turn

        self.prev_space_held = space_held

        # Check for termination conditions
        if self.split_count >= self.WIN_SPLIT_COUNT:
            reward += 100
            terminated = True
        elif self.base_speed < 1.0:
            reward -= 100
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            truncated = True
            
        # Add event-based rewards from line updates
        reward += self._calculate_reward()

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_aiming_input(self, movement):
        angle_change_fine = 0.5
        angle_change_coarse = 2.0
        if movement == 1: # Up
            self.launch_angle -= angle_change_fine
        elif movement == 2: # Down
            self.launch_angle += angle_change_fine
        elif movement == 3: # Left
            self.launch_angle -= angle_change_coarse
        elif movement == 4: # Right
            self.launch_angle += angle_change_coarse

    def _launch_line(self):
        start_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20.0])
        angle_rad = math.radians(self.launch_angle)
        velocity = np.array([math.cos(angle_rad), math.sin(angle_rad)]) * self.base_speed
        
        line = {
            "pos": start_pos,
            "vel": velocity,
            "length": 50,
            "id": self.np_random.integers(1, 1e6)
        }
        self.lines.append(line)

    def _update_lines(self):
        new_lines = []
        collided_lines_ids = set()
        
        for i in range(len(self.lines) -1, -1, -1):
            line = self.lines[i]

            if line["id"] in collided_lines_ids:
                continue

            line["pos"] += line["vel"]

            # Boundary reflection
            start_point = line["pos"]
            end_point = line["pos"] - line["vel"] / np.linalg.norm(line["vel"]) * line["length"]
            
            if not (0 < end_point[0] < self.SCREEN_WIDTH and 0 < end_point[1] < self.SCREEN_HEIGHT):
                if not (0 < start_point[0] < self.SCREEN_WIDTH):
                    line["vel"][0] *= -1
                if not (0 < start_point[1] < self.SCREEN_HEIGHT):
                    line["vel"][1] *= -1
            
            # Off-screen removal
            if not self.screen.get_rect().collidepoint(line["pos"]):
                self.lines.pop(i)
                continue

            # Collision with obstacles
            collided = False
            for obstacle in self.obstacles:
                collision_point = self._line_segment_circle_collision(line, obstacle)
                if collision_point is not None:
                    self.score += 1
                    self.split_count += 1
                    self.split_this_turn = True
                    self.base_speed *= 1.15 # Increase base speed for next launch
                    
                    self._spawn_particles(collision_point, 30)
                    
                    # Create two new lines
                    for _ in range(2):
                        angle_rad = math.atan2(line["vel"][1], line["vel"][0])
                        angle_dev = self.np_random.uniform(-math.pi/4, math.pi/4) # +/- 45 degrees
                        new_angle = angle_rad + angle_dev
                        
                        new_vel = np.array([math.cos(new_angle), math.sin(new_angle)]) * self.base_speed
                        
                        new_line = {
                            "pos": collision_point.copy(),
                            "vel": new_vel,
                            "length": max(20, line["length"] * 0.8),
                            "id": self.np_random.integers(1, 1e6)
                        }
                        new_lines.append(new_line)
                    
                    collided_lines_ids.add(line["id"])
                    self.lines.pop(i)
                    self._spawn_obstacles(1)
                    collided = True
                    break 
            if collided:
                continue
        
        self.lines.extend(new_lines)

    def _line_segment_circle_collision(self, line, circle):
        p1 = line["pos"] - line["vel"] / np.linalg.norm(line["vel"]) * line["length"]
        p2 = line["pos"]
        c_pos = circle["pos"]
        r = circle["radius"]

        d = p2 - p1
        f = p1 - c_pos
        
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c_quad = np.dot(f, f) - r**2
        
        discriminant = b**2 - 4*a*c_quad
        if discriminant < 0:
            return None
        
        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)
        
        if 0 <= t1 <= 1:
            return p1 + t1 * d
        if 0 <= t2 <= 1:
            return p1 + t2 * d
            
        return None

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.pop(i)

    def _spawn_particles(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            life = self.np_random.integers(10, 30)
            self.particles.append({"pos": pos.copy(), "vel": vel, "life": life})

    def _spawn_obstacles(self, count):
        for _ in range(count):
            for _ in range(100): # Max 100 attempts to place
                radius = self.np_random.uniform(10, 25)
                pos = np.array([
                    self.np_random.uniform(radius, self.SCREEN_WIDTH - radius),
                    self.np_random.uniform(radius, self.SCREEN_HEIGHT - radius - 50) # Avoid spawn near player
                ])
                
                # Check for overlap with existing obstacles
                is_overlapping = False
                for obs in self.obstacles:
                    dist = np.linalg.norm(pos - obs["pos"])
                    if dist < radius + obs["radius"] + 10: # 10px buffer
                        is_overlapping = True
                        break
                
                if not is_overlapping:
                    self.obstacles.append({"pos": pos, "radius": radius})
                    break

    def _calculate_reward(self):
        # Rewards are handled directly in the step logic for clarity
        return 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw obstacles
        for obs in self.obstacles:
            pos_int = (int(obs["pos"][0]), int(obs["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(obs["radius"]), self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(obs["radius"]), self.COLOR_OBSTACLE)

        # Draw particles
        for p in self.particles:
            pos_int = (int(p["pos"][0]), int(p["pos"][1]))
            alpha = int(255 * (p["life"] / 30.0))
            color = (*self.COLOR_PARTICLE, alpha)
            temp_surface = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, color, (2, 2), 2)
            self.screen.blit(temp_surface, (pos_int[0]-2, pos_int[1]-2), special_flags=pygame.BLEND_RGBA_ADD)


        # Draw lines with glow
        for line in self.lines:
            end_pos = line["pos"]
            start_pos = end_pos - line["vel"] / np.linalg.norm(line["vel"]) * line["length"]
            self._render_glow_line(self.screen, self.COLOR_LINE, start_pos, end_pos, 3)

        # Draw aiming indicator
        if self.game_phase == 'aiming':
            start_pos = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20)
            angle_rad = math.radians(self.launch_angle)
            end_pos = (start_pos[0] + 60 * math.cos(angle_rad), start_pos[1] + 60 * math.sin(angle_rad))
            
            temp_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.aaline(temp_surface, self.COLOR_AIM, start_pos, end_pos, 2)
            self.screen.blit(temp_surface, (0,0))
            pygame.draw.circle(self.screen, self.COLOR_LINE, (int(start_pos[0]), int(start_pos[1])), 5)

    def _render_glow_line(self, surface, color, start, end, width):
        # Draw a thick, semi-transparent line for the glow
        glow_color = (*color, 30) # Add alpha
        glow_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        pygame.draw.line(glow_surface, glow_color, start, end, width * 4)
        pygame.draw.line(glow_surface, glow_color, start, end, width * 2)
        surface.blit(glow_surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Draw the main anti-aliased line
        pygame.draw.aaline(surface, color, start, end, width-1)

    def _render_ui(self):
        split_text = self.font_large.render(f"Splits: {self.split_count}/{self.WIN_SPLIT_COUNT}", True, self.COLOR_UI)
        self.screen.blit(split_text, (10, 10))

        speed_text = self.font_large.render(f"Speed: {self.base_speed:.1f}", True, self.COLOR_UI)
        self.screen.blit(speed_text, (self.SCREEN_WIDTH - speed_text.get_width() - 10, 10))
        
        if self.game_phase == 'aiming':
            angle_text = self.font_small.render(f"Angle: {self.launch_angle % 360:.1f}°", True, self.COLOR_UI)
            self.screen.blit(angle_text, (self.SCREEN_WIDTH / 2 - angle_text.get_width() / 2, self.SCREEN_HEIGHT - 45))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "split_count": self.split_count,
            "base_speed": self.base_speed,
            "game_phase": self.game_phase
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


# Example usage for interactive play
if __name__ == '__main__':
    # Un-comment the line below to run with a display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Recursive Line Split")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        # --- Pygame event handling for human play ---
        action = [0, 0, 0] # Default no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
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
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Limit to 60 FPS for human play
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Splits: {info['split_count']}")
            # Reset for another game
            obs, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0

    env.close()