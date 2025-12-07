import gymnasium as gym
import os
import pygame
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent transforms between a bird and a mouse
    to navigate a 2D landscape, avoid a patrolling owl, and reach its nest.

    **Visuals:**
    - Player: Bright orange, transforms between a V-shaped bird and a semi-circle mouse.
    - Owl: White circle with a pulsating yellow vision cone.
    - Obstacles: Brown blocks that can hide the mouse.
    - Nest: The green goal square.
    - Background: A dark, starry night sky.

    **Gameplay:**
    - The agent must reach the green nest within the time limit.
    - The owl patrols and will spot the player if they are in its vision cone
      without an obstacle blocking the line of sight.
    - The bird form is fast and can fly over short obstacles.
    - The mouse form is slow but can hide behind obstacles from the owl.
    - Obstacles are added to the level over time, increasing difficulty.

    **Actions:** MultiDiscrete([5, 2, 2])
    - `actions[0]`: Movement (0:none, 1:up, 2:down, 3:left, 4:right)
    - `actions[1]`: Space button (0:released, 1:held) -> Transforms to Bird
    - `actions[2]`: Shift button (0:released, 1:held) -> Transforms to Mouse

    **Rewards:**
    - +100 for reaching the nest.
    - -10 for being detected by the owl.
    - Small positive/negative reward for moving closer/further from the nest.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Transform between a bird and a mouse to navigate a landscape, avoid a patrolling owl, and reach your nest."
    )
    user_guide = (
        "Controls: Use ↑↓←→ to move. Hold Space to transform into a bird (fast, can fly over low obstacles). "
        "Hold Shift to transform into a mouse (slow, can hide)."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 45 * self.FPS  # 45-second game

        # Colors
        self.COLOR_BG_TOP = (15, 25, 50)
        self.COLOR_BG_BOTTOM = (5, 10, 20)
        self.COLOR_PLAYER = (255, 150, 0)
        self.COLOR_PLAYER_GLOW = (255, 180, 50)
        self.COLOR_NEST = (50, 200, 50)
        self.COLOR_OBSTACLE = (139, 69, 19)
        self.COLOR_OBSTACLE_TOP = (160, 82, 45)
        self.COLOR_OWL = (240, 240, 240)
        self.COLOR_OWL_EYE = (10, 10, 10)
        self.COLOR_OWL_VISION = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TIMER_WARN = (255, 100, 100)
        self.COLOR_PARTICLE = (200, 200, 255)

        # Player settings
        self.BIRD_SPEED = 4.0
        self.MOUSE_SPEED = 2.0
        self.BIRD_FLIGHT_CLEARANCE = 40
        self.PLAYER_RADIUS = 10

        # Owl settings
        self.OWL_PATROL_Y = 50
        self.OWL_PATROL_MARGIN = 50
        self.OWL_SPEED = 1.5
        self.OWL_VISION_ANGLE = 45  # degrees
        self.OWL_VISION_RANGE = 200

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_form = 'bird' # 'bird' or 'mouse'
        self.nest_pos = np.array([0, 0])
        self.nest_rect = pygame.Rect(0,0,0,0)
        self.obstacles = []
        self.owl_pos = np.array([0.0, 0.0])
        self.owl_direction = 1
        self.particles = []
        self.stars = []
        self.last_detection_flash = -100

        # self.reset() is called by the test harness, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_detection_flash = -100

        # Place nest and player
        self.nest_pos = np.array([self.WIDTH - 50, self.HEIGHT - 50])
        self.nest_rect = pygame.Rect(self.nest_pos[0] - 15, self.nest_pos[1] - 15, 30, 30)
        self.player_pos = np.array([50.0, 50.0])
        self.player_form = 'bird'

        # Initialize owl
        self.owl_pos = np.array([float(self.OWL_PATROL_MARGIN), float(self.OWL_PATROL_Y)])
        self.owl_direction = 1

        # Generate initial obstacles
        self.obstacles = []
        for _ in range(5):
            self._spawn_obstacle()

        # Generate background stars
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3), self.np_random.uniform(0.1, 1.0))
            for _ in range(100)
        ]
        
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Calculate distance-based reward ---
        dist_before = np.linalg.norm(self.player_pos - self.nest_pos)

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_transformation(space_held, shift_held)
        self._handle_movement(movement)
        
        # --- Update Game Logic ---
        self._update_owl()
        self._update_particles()
        
        # Spawn new obstacles periodically
        if self.steps > 0 and self.steps % (20 * self.FPS) == 0:
            self._spawn_obstacle()
            
        # --- Check for Events and Termination ---
        terminated = False
        
        # 1. Reached Nest (Win)
        if self.nest_rect.collidepoint(self.player_pos):
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = True
        
        # 2. Owl Detection (Lose)
        if self._is_player_detected():
            reward -= 10
            self.score -= 10
            terminated = True
            self.game_over = True
            self.last_detection_flash = self.steps
            
        # 3. Time's Up (Lose)
        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        # Finalize reward calculation
        dist_after = np.linalg.norm(self.player_pos - self.nest_pos)
        reward += (dist_before - dist_after) * 0.1 # Reward for getting closer
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
        
    def _handle_transformation(self, space_held, shift_held):
        old_form = self.player_form
        if space_held:
            self.player_form = 'bird'
        elif shift_held:
            self.player_form = 'mouse'

        if self.player_form != old_form:
            # Transformation particle effect
            for _ in range(20):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
                self.particles.append([self.player_pos.copy(), vel, self.np_random.integers(5, 11)])

    def _handle_movement(self, movement):
        speed = self.BIRD_SPEED if self.player_form == 'bird' else self.MOUSE_SPEED
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] -= speed # Up
        elif movement == 2: move_vec[1] += speed # Down
        elif movement == 3: move_vec[0] -= speed # Left
        elif movement == 4: move_vec[0] += speed # Right

        if np.any(move_vec):
            new_pos = self.player_pos + move_vec
            player_rect = pygame.Rect(new_pos[0] - self.PLAYER_RADIUS, new_pos[1] - self.PLAYER_RADIUS, self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)

            # Obstacle collision
            collided = False
            for obs in self.obstacles:
                if obs.colliderect(player_rect):
                    if self.player_form == 'mouse':
                        collided = True
                        break
                    elif self.player_form == 'bird' and obs.height > self.BIRD_FLIGHT_CLEARANCE:
                        collided = True
                        break
            
            if not collided:
                self.player_pos = new_pos

        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

    def _update_owl(self):
        self.owl_pos[0] += self.owl_direction * self.OWL_SPEED
        if self.owl_pos[0] >= self.WIDTH - self.OWL_PATROL_MARGIN or self.owl_pos[0] <= self.OWL_PATROL_MARGIN:
            self.owl_direction *= -1

    def _is_player_detected(self):
        if self.player_form == 'mouse':
            player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_RADIUS, self.player_pos[1] - self.PLAYER_RADIUS, self.PLAYER_RADIUS*2, self.PLAYER_RADIUS*2)
            for obs in self.obstacles:
                if obs.contains(player_rect):
                    return False # Mouse is hidden inside an obstacle

        if np.linalg.norm(self.player_pos - self.owl_pos) > self.OWL_VISION_RANGE:
            return False

        player_vec = self.player_pos - self.owl_pos
        player_angle = math.degrees(math.atan2(player_vec[1], player_vec[0]))
        
        owl_angle = 0 if self.owl_direction == 1 else 180
        
        angle_diff = abs((player_angle - owl_angle + 180) % 360 - 180)
        
        if angle_diff > self.OWL_VISION_ANGLE / 2:
            return False

        for obs in self.obstacles:
            if obs.clipline(self.owl_pos.tolist(), self.player_pos.tolist()):
                return False

        return True
    
    def _spawn_obstacle(self):
        for _ in range(100):
            w = self.np_random.integers(30, 81)
            h = self.np_random.integers(30, 101)
            x = self.np_random.integers(0, self.WIDTH - w)
            y = self.np_random.integers(self.OWL_PATROL_Y + 30, self.HEIGHT - h)
            new_obs = pygame.Rect(x, y, w, h)

            if new_obs.colliderect(self.nest_rect.inflate(20,20)): continue
            if new_obs.colliderect(pygame.Rect(self.player_pos[0]-15, self.player_pos[1]-15, 30, 30)): continue
            
            is_overlapping = any(new_obs.colliderect(obs) for obs in self.obstacles)
            if not is_overlapping:
                self.obstacles.append(new_obs)
                return
    
    def _update_particles(self):
        self.particles = [
            [p[0] + p[1], p[1] * 0.95, p[2] - 0.3]
            for p in self.particles if p[2] > 0
        ]
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG_BOTTOM)
        self._render_background_gradient()
        self._render_stars()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_gradient(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio,
                self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio,
                self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_stars(self):
        for x, y, r, alpha_mod in self.stars:
            alpha = abs(math.sin(self.steps * 0.05 + x)) * 255 * alpha_mod
            color = (255, 255, 255, int(alpha))
            temp_surf = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (r,r), r)
            self.screen.blit(temp_surf, (x-r, y-r))

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_NEST, self.nest_rect)
        pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_NEST), self.nest_rect, 3)
        
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_TOP, (obs.x, obs.y, obs.width, 5))

        self._render_owl()
        self._render_particles()
        self._render_player()

    def _render_owl(self):
        cone_pulse = 1.0 + 0.05 * math.sin(self.steps * 0.2)
        range_pulse = self.OWL_VISION_RANGE * cone_pulse
        
        angle_rad = math.radians(self.OWL_VISION_ANGLE / 2)
        dir_angle = 0 if self.owl_direction == 1 else math.pi
        
        p1 = self.owl_pos
        p2 = self.owl_pos + np.array([math.cos(dir_angle - angle_rad), math.sin(dir_angle - angle_rad)]) * range_pulse
        p3 = self.owl_pos + np.array([math.cos(dir_angle + angle_rad), math.sin(dir_angle + angle_rad)]) * range_pulse
        
        vision_poly = [(int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1]))]
        
        temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(temp_surf, vision_poly, (*self.COLOR_OWL_VISION, 60))
        pygame.gfxdraw.aapolygon(temp_surf, vision_poly, (*self.COLOR_OWL_VISION, 120))
        self.screen.blit(temp_surf, (0,0))
        
        ox, oy = int(self.owl_pos[0]), int(self.owl_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, ox, oy, 15, self.COLOR_OWL)
        pygame.gfxdraw.aacircle(self.screen, ox, oy, 15, self.COLOR_OWL)
        eye_x_offset = 5 * self.owl_direction
        pygame.draw.circle(self.screen, self.COLOR_OWL_EYE, (ox + eye_x_offset, oy - 2), 3)
        pygame.draw.circle(self.screen, self.COLOR_OWL_EYE, (ox - eye_x_offset, oy - 2), 3)

    def _render_player(self):
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        
        temp_surf = pygame.Surface((self.PLAYER_RADIUS*2+10, self.PLAYER_RADIUS*2+10), pygame.SRCALPHA)
        glow_center = temp_surf.get_width() // 2, temp_surf.get_height() // 2
        for i in range(5):
            alpha = 100 - i * 20
            radius = self.PLAYER_RADIUS + i * 2
            pygame.draw.circle(temp_surf, (*self.COLOR_PLAYER_GLOW, alpha), glow_center, radius)
        self.screen.blit(temp_surf, (px - glow_center[0], py - glow_center[1]))

        if self.player_form == 'bird':
            flap_angle = math.radians(20 + 10 * math.sin(self.steps * 0.5))
            p1 = (px + self.PLAYER_RADIUS * math.cos(math.pi + flap_angle), py + self.PLAYER_RADIUS * math.sin(math.pi + flap_angle))
            p2 = (px, py)
            p3 = (px + self.PLAYER_RADIUS * math.cos(math.pi - flap_angle), py + self.PLAYER_RADIUS * math.sin(math.pi - flap_angle))
            pygame.draw.aalines(self.screen, self.COLOR_PLAYER, False, [p1,p2,p3], 3)
        else: # mouse
            # Draw a filled semi-circle for the mouse body
            points = [(px, py)]
            for angle_deg in range(180, 361):
                angle_rad = math.radians(angle_deg)
                x = px + self.PLAYER_RADIUS * math.cos(angle_rad)
                y = py + self.PLAYER_RADIUS * math.sin(angle_rad)
                points.append((x, y))
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)

    def _render_particles(self):
        for pos, vel, life in self.particles:
            px, py = int(pos[0]), int(pos[1])
            radius = int(life / 2)
            if radius > 0:
                pygame.draw.circle(self.screen, self.COLOR_PARTICLE, (px, py), radius)
    
    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_color = self.COLOR_TIMER_WARN if time_left < 10 else self.COLOR_TEXT
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        if self.game_over:
            is_win = not self.game_over or self.nest_rect.collidepoint(self.player_pos)
            if self.steps - self.last_detection_flash < self.FPS * 0.5:
                overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                overlay.fill((255,0,0,100))
                self.screen.blit(overlay, (0,0))
            
            result_text_str = "NEST REACHED!" if is_win and self.steps < self.MAX_STEPS else "GAME OVER"
            result_text = self.font_game_over.render(result_text_str, True, self.COLOR_TEXT)
            text_rect = result_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(result_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_form": self.player_form,
            "time_left_seconds": max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Use a real display for manual testing
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bird/Mouse Transformation Game")
    clock = pygame.time.Clock()
    
    done = False
    
    movement_action = 0
    space_action = 0
    shift_action = 0
    
    print("\n--- Manual Control ---")
    print("ARROWS: Move")
    print("SPACE: Transform to Bird (Hold)")
    print("SHIFT: Transform to Mouse (Hold)")
    print("Q: Quit")
    print("R: Reset")
    print("----------------------\n")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("--- Environment Reset ---")

        keys = pygame.key.get_pressed()
        
        movement_action = 0
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
            
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(env.FPS)
        
    env.close()