
# Generated: 2025-08-28T04:50:38.362742
# Source Brief: brief_02431.md
# Brief Index: 2431

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    An arcade game where the player launches balls to demolish a procedurally generated castle.
    The goal is to destroy all bricks within a limited number of shots.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑/↓ to adjust angle, ←/→ to adjust power. Press space to launch the ball."
    )

    game_description = (
        "Launch balls to demolish a castle of colorful bricks. Destroy all bricks before you run out of balls!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Colors
        self.COLOR_BG_TOP = (40, 40, 60)
        self.COLOR_BG_BOTTOM = (10, 10, 20)
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_TRAJECTORY = (255, 255, 255, 100)
        self.BRICK_COLORS = {
            10: (220, 80, 80),  # Red
            20: (80, 220, 80),  # Green
            30: (80, 80, 220),  # Blue
        }
        self.COLOR_BALL = (255, 200, 0)
        self.COLOR_SHADOW = (0, 0, 0, 50)

        # Game constants
        self.MAX_STEPS = 1000
        self.INITIAL_BALLS = 5
        self.NUM_BRICKS = 75
        self.GRAVITY = 0.08

        # Isometric projection constants
        self.ISO_X_AXIS = pygame.Vector2(0.866, -0.5) * 12
        self.ISO_Y_AXIS = pygame.Vector2(0.866, 0.5) * 12
        self.ISO_Z_AXIS = pygame.Vector2(0, -1) * 12
        self.ISO_ORIGIN = pygame.Vector2(self.screen_width // 2, self.screen_height - 50)
        
        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_remaining = 0
        self.game_phase = "aiming"  # "aiming", "firing", "result"
        self.bricks = []
        self.ball = None
        self.particles = []
        self.aim_angle = 0.0
        self.aim_power = 0.0
        self.last_shot_info = {"hits": 0, "score": 0}

        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def _generate_castle(self):
        """Procedurally generates a castle structure with 75 bricks."""
        self.bricks = []
        bricks_to_place = self.NUM_BRICKS
        
        # Base layer
        base_size = 6
        for y in range(base_size):
            for x in range(base_size):
                if bricks_to_place > 0:
                    self.bricks.append(self._create_brick(x, y, 0))
                    bricks_to_place -= 1
        
        # Second layer with gaps
        for y in range(1, base_size - 1):
            for x in range(1, base_size - 1):
                 if bricks_to_place > 0:
                    self.bricks.append(self._create_brick(x, y, 1))
                    bricks_to_place -= 1

        # Towers
        tower_height = 4
        for z in range(2, tower_height + 2):
            if bricks_to_place > 0: self.bricks.append(self._create_brick(0, 0, z)); bricks_to_place -=1
            if bricks_to_place > 0: self.bricks.append(self._create_brick(base_size-1, 0, z)); bricks_to_place -=1
            if bricks_to_place > 0: self.bricks.append(self._create_brick(0, base_size-1, z)); bricks_to_place -=1
            if bricks_to_place > 0: self.bricks.append(self._create_brick(base_size-1, base_size-1, z)); bricks_to_place -=1

        # Fill remaining bricks randomly on top
        while bricks_to_place > 0:
            x = self.np_random.integers(0, base_size)
            y = self.np_random.integers(0, base_size)
            z = self.np_random.integers(2, 6)
            # Avoid placing floating bricks
            has_support = any(b['pos'] == (x,y,z-1) for b in self.bricks)
            is_occupied = any(b['pos'] == (x,y,z) for b in self.bricks)
            if has_support and not is_occupied:
                self.bricks.append(self._create_brick(x, y, z))
                bricks_to_place -= 1
        
        # Sort for rendering
        self.bricks.sort(key=lambda b: b['pos'][0] + b['pos'][1] - b['pos'][2])

    def _create_brick(self, x, y, z):
        """Creates a single brick dictionary."""
        points = self.np_random.choice([10, 20, 30])
        color = self.BRICK_COLORS[points]
        return {
            "pos": (x, y, z),
            "points": points,
            "color": color,
            "destroyed": False
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.balls_remaining = self.INITIAL_BALLS
        self.game_phase = "aiming"
        self.ball = None
        self.particles = []
        self.aim_angle = 45.0
        self.aim_power = 5.0
        self.last_shot_info = {"hits": 0, "score": 0}

        self._generate_castle()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, _ = action
        reward = 0.0
        
        self._update_particles()

        if self.game_phase == "aiming":
            self._handle_aiming(movement, space_pressed)
        elif self.game_phase == "firing":
            reward = self._handle_firing()
        elif self.game_phase == "result":
            self.game_phase = "aiming"

        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            destroyed_bricks = sum(1 for b in self.bricks if b['destroyed'])
            if destroyed_bricks == len(self.bricks):
                reward += 100  # Win bonus
            else:
                reward -= 100  # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_aiming(self, movement, space_pressed):
        """Updates aim based on player input."""
        # up/down to adjust angle
        if movement == 1: self.aim_angle += 1.0
        if movement == 2: self.aim_angle -= 1.0
        # left/right to adjust power
        if movement == 3: self.aim_power -= 0.1
        if movement == 4: self.aim_power += 0.1
        
        self.aim_angle = np.clip(self.aim_angle, 15, 85)
        self.aim_power = np.clip(self.aim_power, 2, 8)

        if space_pressed and self.balls_remaining > 0:
            self.balls_remaining -= 1
            self.game_phase = "firing"
            angle_rad = math.radians(self.aim_angle)
            # Launch towards the center of the castle
            vel_x = math.cos(angle_rad) * self.aim_power * 0.5
            vel_y = math.cos(angle_rad) * self.aim_power * 0.5
            vel_z = math.sin(angle_rad) * self.aim_power
            self.ball = {
                "pos": pygame.Vector3(0, 0, 0),
                "vel": pygame.Vector3(vel_x, vel_y, vel_z)
            }
            # Sound: Launch ball

    def _handle_firing(self):
        """Updates ball physics and checks for collisions."""
        if self.ball is None:
            self.game_phase = "result"
            return 0.0

        # Update physics
        self.ball["vel"].z -= self.GRAVITY
        self.ball["pos"] += self.ball["vel"] * 0.2  # Time scaling

        shot_reward = 0.0
        hit_this_step = False
        
        # Check collision with bricks
        for brick in self.bricks:
            if not brick["destroyed"]:
                b_pos = pygame.Vector3(brick["pos"])
                if b_pos.distance_to(self.ball["pos"]) < 1.0:
                    brick["destroyed"] = True
                    self.last_shot_info["hits"] += 1
                    self.last_shot_info["score"] += brick["points"]
                    self._create_explosion(b_pos, brick["color"])
                    hit_this_step = True
                    # Sound: Brick shatter

        # If ball hits anything or goes out of bounds, end the turn
        if hit_this_step or self.ball["pos"].z < -1:
            self.game_phase = "result"
            self.score += self.last_shot_info["score"]
            
            if self.last_shot_info["hits"] > 0:
                shot_reward = 1.0 + (self.last_shot_info["score"] / 10.0)
            else:
                shot_reward = -0.1 # Miss penalty
                # Sound: Ball thud
            
            self.ball = None
            # Reset for next shot
            self.last_shot_info = {"hits": 0, "score": 0}
            return shot_reward
        
        return 0.0

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        destroyed_bricks = sum(1 for b in self.bricks if b['destroyed'])
        if destroyed_bricks == len(self.bricks):
            return True # Win
        if self.balls_remaining == 0 and self.game_phase == "aiming":
            return True # Loss
        return False

    def _get_observation(self):
        self._draw_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_background(self):
        """Draws a vertical gradient background."""
        for y in range(self.screen_height):
            ratio = y / self.screen_height
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))

    def _render_game(self):
        # Render bricks
        for brick in self.bricks:
            if not brick["destroyed"]:
                self._draw_iso_cube(brick["pos"], brick["color"])
        
        # Render aiming trajectory
        if self.game_phase == "aiming":
            self._draw_trajectory()
        
        # Render ball and shadow
        if self.ball:
            # Shadow
            shadow_pos_3d = pygame.Vector3(self.ball["pos"].x, self.ball["pos"].y, 0)
            shadow_pos_2d = self._iso_to_screen(shadow_pos_3d)
            shadow_size = max(2, 10 - self.ball["pos"].z * 0.5)
            shadow_surface = pygame.Surface((shadow_size*2, shadow_size*2), pygame.SRCALPHA)
            pygame.draw.circle(shadow_surface, self.COLOR_SHADOW, (shadow_size, shadow_size), shadow_size)
            self.screen.blit(shadow_surface, (shadow_pos_2d.x - shadow_size, shadow_pos_2d.y - shadow_size))

            # Ball
            ball_pos_2d = self._iso_to_screen(self.ball["pos"])
            pygame.gfxdraw.filled_circle(self.screen, int(ball_pos_2d.x), int(ball_pos_2d.y), 6, self.COLOR_BALL)
            pygame.gfxdraw.aacircle(self.screen, int(ball_pos_2d.x), int(ball_pos_2d.y), 6, self.COLOR_WHITE)

        # Render particles
        for p in self.particles:
            p_pos_2d = self._iso_to_screen(p["pos"])
            alpha = max(0, 255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (p_pos_2d.x - p['size'], p_pos_2d.y - p['size']))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))

        # Balls remaining
        balls_text = self.font_small.render(f"BALLS: {self.balls_remaining}", True, self.COLOR_WHITE)
        self.screen.blit(balls_text, (self.screen_width - balls_text.get_width() - 10, 10))
        
        # Aiming UI
        if self.game_phase == "aiming":
            angle_text = self.font_small.render(f"Angle: {self.aim_angle:.1f}", True, self.COLOR_WHITE)
            power_text = self.font_small.render(f"Power: {self.aim_power:.1f}", True, self.COLOR_WHITE)
            self.screen.blit(angle_text, (10, self.screen_height - 40))
            self.screen.blit(power_text, (10, self.screen_height - 25))

        # Game Over message
        if self.game_over:
            destroyed_bricks = sum(1 for b in self.bricks if b['destroyed'])
            if destroyed_bricks == len(self.bricks):
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            text = self.font_large.render(msg, True, color)
            text_rect = text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "balls_remaining": self.balls_remaining}

    def _iso_to_screen(self, pos_3d):
        """Converts 3D isometric coordinates to 2D screen coordinates."""
        return self.ISO_ORIGIN + pos_3d.x * self.ISO_X_AXIS + pos_3d.y * self.ISO_Y_AXIS + pos_3d.z * self.ISO_Z_AXIS

    def _draw_iso_cube(self, pos, color):
        """Draws a 3D-looking cube in isometric projection."""
        x, y, z = pos
        top = self._iso_to_screen(pygame.Vector3(x, y, z + 1))
        p = [
            self._iso_to_screen(pygame.Vector3(x, y, z)),
            self._iso_to_screen(pygame.Vector3(x + 1, y, z)),
            self._iso_to_screen(pygame.Vector3(x + 1, y + 1, z)),
            self._iso_to_screen(pygame.Vector3(x, y + 1, z)),
        ]
        
        darker_color = tuple(max(0, c - 50) for c in color)
        darkest_color = tuple(max(0, c - 80) for c in color)
        
        # Draw faces with polygons for solid look
        pygame.draw.polygon(self.screen, color, [(p[3].x, p[3].y), (p[2].x, p[2].y), (top.x, top.y + self.ISO_Z_AXIS.y), (top.x, top.y)]) # Right face
        pygame.draw.polygon(self.screen, darker_color, [(p[0].x, p[0].y), (p[3].x, p[3].y), (top.x, top.y), (top.x - self.ISO_X_AXIS.x, top.y - self.ISO_X_AXIS.y)]) # Left face
        pygame.draw.polygon(self.screen, darkest_color, [(top.x - self.ISO_X_AXIS.x, top.y - self.ISO_X_AXIS.y), (top.x, top.y), (top.x, top.y + self.ISO_Z_AXIS.y), (top.x - self.ISO_Y_AXIS.x, top.y - self.ISO_Y_AXIS.y)]) # Top face
        
        # Draw outlines for definition
        pygame.draw.aaline(self.screen, self.COLOR_WHITE, p[0], p[1], 2)
        pygame.draw.aaline(self.screen, self.COLOR_WHITE, p[1], p[2], 2)
        pygame.draw.aaline(self.screen, self.COLOR_WHITE, p[2], p[3], 2)
        pygame.draw.aaline(self.screen, self.COLOR_WHITE, p[3], p[0], 2)

    def _draw_trajectory(self):
        """Simulates and draws the ball's predicted path."""
        angle_rad = math.radians(self.aim_angle)
        vel_x = math.cos(angle_rad) * self.aim_power * 0.5
        vel_y = math.cos(angle_rad) * self.aim_power * 0.5
        vel_z = math.sin(angle_rad) * self.aim_power
        
        pos = pygame.Vector3(0, 0, 0)
        vel = pygame.Vector3(vel_x, vel_y, vel_z)

        for _ in range(60): # Simulate 60 steps
            vel.z -= self.GRAVITY
            pos += vel * 0.2
            if _ % 3 == 0: # Draw a dot every 3 steps
                screen_pos = self._iso_to_screen(pos)
                pygame.draw.circle(self.screen, self.COLOR_TRAJECTORY, (int(screen_pos.x), int(screen_pos.y)), 2)
            if pos.z < -1:
                break
    
    def _create_explosion(self, pos_3d, color):
        """Generates particles for a brick explosion."""
        for _ in range(20):
            vel = pygame.Vector3(
                (self.np_random.random() - 0.5) * 1.5,
                (self.np_random.random() - 0.5) * 1.5,
                (self.np_random.random()) * 1.5
            )
            life = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": pos_3d.copy(), "vel": vel, "life": life, "max_life": life,
                "color": color, "size": self.np_random.integers(2, 5)
            })

    def _update_particles(self):
        """Updates position and lifetime of all particles."""
        for p in self.particles:
            p["vel"].z -= self.GRAVITY * 0.5
            p["pos"] += p["vel"] * 0.2
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        assert len(self.bricks) > 0, "Castle generation failed."
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part is for human testing and requires a display.
    # It will not run in a headless environment.
    try:
        import os
        os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        # Set up a window for human play
        screen = pygame.display.set_mode((env.screen_width, env.screen_height))
        pygame.display.set_caption("Castle Crusher")
        
        obs, info = env.reset()
        done = False
        
        while not done:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_SHIFT]: shift = 1

            action = [movement, space, 0] # Shift is unused
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render to the display window
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Since auto_advance is False, we need to control the speed of the game loop for human play.
            # Aiming phase can be instant, but firing phase needs a delay to be visible.
            if env.game_phase == "firing":
                pygame.time.wait(30) # ~33 FPS for ball flight
            else:
                pygame.time.wait(10) # Small delay to prevent high CPU usage

        print(f"Game Over! Final Score: {info['score']}")
        pygame.time.wait(2000) # Wait 2 seconds before closing
    
    env.close()