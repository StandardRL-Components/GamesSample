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
    """
    A Gymnasium environment where the player draws lines to guide a sled down a procedurally generated slope.
    The goal is to reach the finish line while balancing speed and stability.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to aim your line-drawer. Hold Space to draw a path for your sled."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Draw lines to guide your sled down a procedurally generated slope, balancing speed and stability to reach the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

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
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)

        # Game Constants
        self.MAX_STEPS = 900  # 30 seconds at 30fps
        self.GRAVITY = pygame.math.Vector2(0, 0.25)
        self.SLED_LENGTH = 20
        self.LINE_LENGTH = 30
        self.MAX_LINES = 50
        self.AIMER_RADIUS = 80
        self.AIMER_TURN_RATE = 0.1

        # Colors
        self.COLOR_BG_TOP = (20, 25, 40)
        self.COLOR_BG_BOTTOM = (40, 50, 80)
        self.COLOR_SLED = (255, 255, 255)
        self.COLOR_SLED_GLOW = (200, 200, 255, 64)
        self.COLOR_LINE = (0, 192, 255)
        self.COLOR_TERRAIN = (128, 140, 160)
        self.COLOR_START = (0, 255, 128)
        self.COLOR_FINISH = (255, 80, 80)
        self.COLOR_CHECKPOINT = (255, 224, 0)
        self.COLOR_AIMER = (255, 255, 255, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_PARTICLE = (255, 224, 0)

        # Initialize state variables
        self.sled_pos_front = pygame.math.Vector2(0, 0)
        self.sled_pos_back = pygame.math.Vector2(0, 0)
        self.sled_old_pos_front = pygame.math.Vector2(0, 0)
        self.sled_old_pos_back = pygame.math.Vector2(0, 0)
        self.lines = []
        self.terrain_points = []
        self.checkpoints = []
        self.particles = []
        self.draw_anchor = pygame.math.Vector2(0, 0)
        self.aimer_angle = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_forward_progress = 0

    def _generate_course(self):
        """Creates the procedural terrain, checkpoints, and start/finish lines."""
        self.terrain_points = []
        start_y = self.HEIGHT * 0.25
        
        points = [(0, start_y + self.np_random.uniform(-10, 10))]
        
        num_segments = 15
        for i in range(1, num_segments + 1):
            x = i * self.WIDTH / num_segments
            y = start_y + (i * 20) + self.np_random.uniform(-30, 30)
            points.append((x, y))
        points.append((self.WIDTH + 50, points[-1][1] + 50)) # Ensure terrain extends off-screen

        self.terrain_points = [pygame.math.Vector2(p) for p in points]

        # Calculate start position based on terrain height at x=50
        start_x = 50
        terrain_y_at_start = self.HEIGHT
        for j in range(len(self.terrain_points) - 1):
            p1, p2 = self.terrain_points[j], self.terrain_points[j + 1]
            if p1.x <= start_x <= p2.x:
                if (p2.x - p1.x) != 0:
                    interp = (start_x - p1.x) / (p2.x - p1.x)
                    terrain_y_at_start = p1.y + interp * (p2.y - p1.y)
                else: # Vertical segment
                    terrain_y_at_start = p1.y
                break
        
        self.start_pos = pygame.math.Vector2(start_x, terrain_y_at_start - 10) # Place just above the terrain
        self.finish_line_x = self.terrain_points[-2].x

        self.checkpoints = []
        num_checkpoints = 3
        for i in range(1, num_checkpoints + 1):
            x_pos = self.start_pos.x + i * (self.finish_line_x - self.start_pos.x) / (num_checkpoints + 1)
            
            y_pos = self.HEIGHT
            for j in range(len(self.terrain_points) - 1):
                p1, p2 = self.terrain_points[j], self.terrain_points[j+1]
                if p1.x <= x_pos <= p2.x:
                    interp = (x_pos - p1.x) / (p2.x - p1.x) if (p2.x - p1.x) != 0 else 0
                    y_pos = p1.y + interp * (p2.y - p1.y)
                    break
            
            self.checkpoints.append({
                "pos": pygame.math.Vector2(x_pos, y_pos - 50),
                "radius": 15,
                "reached": False
            })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._generate_course()

        self.sled_pos_back = self.start_pos.copy()
        self.sled_pos_front = self.start_pos + pygame.math.Vector2(self.SLED_LENGTH, 0)
        self.sled_old_pos_back = self.sled_pos_back.copy()
        self.sled_old_pos_front = self.sled_pos_front.copy()

        self.lines = []
        self.particles = []
        self.draw_anchor = self.sled_pos_back.copy()
        self.aimer_angle = 0

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_forward_progress = self.start_pos.x
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if movement == 1: # Up
            self.aimer_angle -= self.AIMER_TURN_RATE
        elif movement == 2: # Down
            self.aimer_angle += self.AIMER_TURN_RATE
        elif movement == 3: # Left
            self.aimer_angle -= self.AIMER_TURN_RATE * 2
        elif movement == 4: # Right
            self.aimer_angle += self.AIMER_TURN_RATE * 2
        
        self.aimer_angle = self.aimer_angle % (2 * math.pi)

        if space_held:
            sled_center = (self.sled_pos_front + self.sled_pos_back) / 2
            if self.draw_anchor.distance_to(sled_center) > self.AIMER_RADIUS * 1.5:
                self.draw_anchor = sled_center.copy()

            new_point = self.draw_anchor + pygame.math.Vector2(self.LINE_LENGTH, 0).rotate_rad(self.aimer_angle)
            
            if len(self.lines) < self.MAX_LINES:
                self.lines.append((self.draw_anchor, new_point))
                # sfx: line draw sound
                self.draw_anchor = new_point
                
                for _ in range(3):
                    self.particles.append(self._create_particle(new_point, self.COLOR_LINE))

    def _update_physics(self):
        # Verlet integration for sled physics
        vel_back = self.sled_pos_back - self.sled_old_pos_back
        vel_front = self.sled_pos_front - self.sled_old_pos_front

        self.sled_old_pos_back = self.sled_pos_back.copy()
        self.sled_old_pos_front = self.sled_pos_front.copy()

        self.sled_pos_back += vel_back + self.GRAVITY
        self.sled_pos_front += vel_front + self.GRAVITY

        all_surfaces = self.lines + list(zip(self.terrain_points[:-1], self.terrain_points[1:]))
        
        # Iteratively solve constraints for stability
        for _ in range(3):
            self._solve_collisions(self.sled_pos_back, self.sled_old_pos_back, all_surfaces)
            self._solve_collisions(self.sled_pos_front, self.sled_old_pos_front, all_surfaces)
            self._solve_stick_constraint()

    def _solve_collisions(self, pos, old_pos, surfaces):
        """Handles collision detection and response for a single point of the sled."""
        for p1, p2 in surfaces:
            line_vec = p2 - p1
            point_vec = pos - p1
            line_len_sq = line_vec.length_squared()
            if line_len_sq == 0: continue

            t = point_vec.dot(line_vec) / line_len_sq
            t = max(0, min(1, t))
            closest_point = p1 + t * line_vec
            
            dist_vec = pos - closest_point
            dist = dist_vec.length()
            
            if dist < 2: # Collision detected
                # sfx: sled scrape/impact sound
                pos += dist_vec.normalize() * (2 - dist)
                
                vel = pos - old_pos
                normal = dist_vec.normalize()
                vel_normal_comp = vel.dot(normal)
                
                if vel_normal_comp < 0: # Moving into the surface
                    vel.reflect_ip(normal)
                    vel *= 0.8 # Restitution (bounce)
                    
                    tangent = pygame.math.Vector2(-normal.y, normal.x)
                    vel_tangent_comp = vel.dot(tangent)
                    vel -= tangent * vel_tangent_comp * 0.1 # Friction
                
                new_old_pos = pos - vel
                old_pos.x, old_pos.y = new_old_pos.x, new_old_pos.y

                if self.np_random.random() < 0.2:
                    self.particles.append(self._create_particle(pos, self.COLOR_PARTICLE))

    def _solve_stick_constraint(self):
        """Ensures the two points of the sled stay a fixed distance apart."""
        center = (self.sled_pos_front + self.sled_pos_back) / 2
        dist_vec = self.sled_pos_front - self.sled_pos_back
        dist = dist_vec.length()
        if dist == 0: return
        
        error = (dist - self.SLED_LENGTH) / dist
        
        self.sled_pos_front -= error * 0.5 * dist_vec
        self.sled_pos_back += error * 0.5 * dist_vec

    def _update_particles(self):
        """Updates position and lifetime of all particles."""
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'] *= 0.95 # Drag

        sled_center = (self.sled_pos_front + self.sled_pos_back) / 2
        sled_vel = sled_center - ((self.sled_old_pos_front + self.sled_old_pos_back) / 2)
        if sled_vel.length() > 5:
            # sfx: whoosh sound
            for _ in range(2):
                start_pos = sled_center + pygame.math.Vector2(self.np_random.uniform(-5, 5), self.np_random.uniform(-5, 5))
                self.particles.append({
                    'pos': start_pos,
                    'vel': -sled_vel.normalize() * self.np_random.uniform(3, 6) + pygame.math.Vector2(self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-0.5, 0.5)),
                    'life': 10, 'color': self.COLOR_CHECKPOINT, 'type': 'line'
                })

    def _create_particle(self, pos, color):
        return {
            'pos': pos.copy(),
            'vel': pygame.math.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
            'life': self.np_random.integers(10, 20),
            'color': color, 'type': 'circle'
        }

    def _calculate_reward(self):
        if self.game_over:
            sled_center = (self.sled_pos_front + self.sled_pos_back) / 2
            if sled_center.x >= self.finish_line_x: return 100.0
            else: return -100.0

        reward = 0.0
        sled_center = (self.sled_pos_front + self.sled_pos_back) / 2
        
        if sled_center.x > self.last_forward_progress:
            reward += 0.1 * (sled_center.x - self.last_forward_progress) / 10.0
            self.last_forward_progress = sled_center.x
        else:
            reward -= 0.01

        for cp in self.checkpoints:
            if not cp["reached"] and sled_center.distance_to(cp["pos"]) < cp["radius"]:
                cp["reached"] = True
                reward += 10.0
                # sfx: checkpoint reached sound
                for _ in range(20):
                    self.particles.append(self._create_particle(cp['pos'], cp['pos'].lerp(self.COLOR_SLED, 0.5)))
        
        return reward

    def _check_termination(self):
        sled_center = (self.sled_pos_front + self.sled_pos_back) / 2
        if not (0 < sled_center.x < self.WIDTH and -50 < sled_center.y < self.HEIGHT + 50):
            self.game_over = True # Out of bounds
        
        if sled_center.x >= self.finish_line_x:
            self.game_over = True # Reached finish
            self.score += 100
        
        if self.sled_pos_front.y > self.sled_pos_back.y + self.SLED_LENGTH * 0.98:
            if not self._is_on_ground(self.sled_pos_front) and not self._is_on_ground(self.sled_pos_back):
                 self.game_over = True # Flipped in mid-air
        
        return self.game_over or self.steps >= self.MAX_STEPS

    def _is_on_ground(self, point):
        """Checks if a point is close to any surface."""
        all_surfaces = self.lines + list(zip(self.terrain_points[:-1], self.terrain_points[1:]))
        for p1, p2 in all_surfaces:
            line_vec = p2 - p1
            point_vec = point - p1
            line_len_sq = line_vec.length_squared()
            if line_len_sq == 0: continue
            t = point_vec.dot(line_vec) / line_len_sq
            t = max(0, min(1, t))
            closest_point = p1 + t * line_vec
            if point.distance_to(closest_point) < 5:
                return True
        return False

    def step(self, action):
        if self.game_over:
            reward = 0.0
            terminated = True
        else:
            self._handle_input(action)
            self._update_physics()
            self._update_particles()
            
            reward = self._calculate_reward()
            self.score += reward
            terminated = self._check_termination()
            self.steps += 1
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self._render_background()
        # Render all game elements
        self._render_game()
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        """Draws a vertical gradient for the background."""
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        """Renders all non-UI game elements."""
        pygame.draw.line(self.screen, self.COLOR_START, (self.start_pos.x, 0), (self.start_pos.x, self.HEIGHT), 2)
        pygame.draw.line(self.screen, self.COLOR_FINISH, (self.finish_line_x, 0), (self.finish_line_x, self.HEIGHT), 2)

        pygame.draw.aalines(self.screen, self.COLOR_TERRAIN, False, [(int(p.x), int(p.y)) for p in self.terrain_points], 2)

        for p1, p2 in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_LINE, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), 2)

        for cp in self.checkpoints:
            color = self.COLOR_SLED if cp["reached"] else self.COLOR_CHECKPOINT
            pygame.gfxdraw.aacircle(self.screen, int(cp["pos"].x), int(cp["pos"].y), int(cp["radius"]), color)
            pygame.gfxdraw.filled_circle(self.screen, int(cp["pos"].x), int(cp["pos"].y), int(cp["radius"] - 2), (*color, 60))

        sled_center = (self.sled_pos_front + self.sled_pos_back) / 2
        aim_anchor = self.draw_anchor
        if aim_anchor.distance_to(sled_center) > self.AIMER_RADIUS * 1.5:
             aim_anchor = sled_center
        aim_target = aim_anchor + pygame.math.Vector2(self.LINE_LENGTH, 0).rotate_rad(self.aimer_angle)
        pygame.draw.aaline(self.screen, self.COLOR_AIMER, (int(aim_anchor.x), int(aim_anchor.y)), (int(aim_target.x), int(aim_target.y)))
        pygame.gfxdraw.aacircle(self.screen, int(aim_target.x), int(aim_target.y), 3, self.COLOR_AIMER)

        for p in self.particles:
            alpha = int(255 * (p['life'] / 20.0))
            color = (*p['color'], alpha)
            if p['type'] == 'circle':
                 pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 2, color)
            elif p['type'] == 'line':
                 end_pos = p['pos'] - p['vel'] * 0.5
                 pygame.draw.aaline(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), (int(end_pos.x), int(end_pos.y)))

        p1 = (int(self.sled_pos_back.x), int(self.sled_pos_back.y))
        p2 = (int(self.sled_pos_front.x), int(self.sled_pos_front.y))
        pygame.gfxdraw.filled_circle(self.screen, int(sled_center.x), int(sled_center.y), int(self.SLED_LENGTH * 1.2), self.COLOR_SLED_GLOW)
        pygame.draw.line(self.screen, self.COLOR_SLED, p1, p2, 5)
        pygame.draw.circle(self.screen, self.COLOR_SLED, p2, 4)

    def _render_ui(self):
        """Renders the score, time, and game over messages."""
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        score_text = self.font_small.render(f"SCORE: {self.score:.0f}", True, self.COLOR_TEXT)
        time_text = self.font_small.render(f"TIME: {max(0, time_left):.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (10, 30))

        if self.game_over:
            sled_center = (self.sled_pos_front + self.sled_pos_back) / 2
            msg = "FINISH!" if sled_center.x >= self.finish_line_x else "CRASHED"
            color = self.COLOR_START if msg == "FINISH!" else self.COLOR_FINISH
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS,
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # The following is not part of the required fixed environment but allows for human play
    # It has been sanitized to use the public API of the environment
    
    # Re-initialize pygame for display
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit()
    pygame.init()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Sled Rider")
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    print(env.game_description)
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default to no-op
        
        # --- Human Controls ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        move_action = 0 # none
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([move_action, space_action, shift_action])
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
        if terminated or truncated:
            print(f"Game Over. Final Score: {info['score']:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            
    env.close()
    pygame.quit()