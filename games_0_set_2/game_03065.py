
# Generated: 2025-08-28T06:52:34.264198
# Source Brief: brief_03065.md
# Brief Index: 3065

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: ↑ to accelerate, ←→ to turn, and ↓ to brake. Hold Shift to drift for tighter turns."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced retro arcade racer. Dodge obstacles and race against the clock to reach the finish line."
    )

    # Frames auto-advance for smooth gameplay and time limits.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME = 20  # seconds
        self.MAX_STEPS = self.MAX_TIME * self.FPS

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (255, 60, 60)
        self.COLOR_PLAYER_GLOW = (180, 40, 40)
        self.COLOR_OBSTACLE = (60, 120, 255)
        self.COLOR_OBSTACLE_GLOW = (40, 80, 180)
        self.COLOR_FINISH_LINE = (255, 255, 255)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_DRIFT_SMOKE = (150, 150, 150)
        self.COLOR_EXPLOSION = (255, 255, 255)

        # Player Physics
        self.ACCELERATION = 0.4
        self.BRAKING = 0.8
        self.FRICTION = 0.96
        self.MAX_SPEED = 8.0
        self.TURN_SPEED = 0.08
        self.DRIFT_TURN_MOD = 2.0
        self.DRIFT_FRICTION_MOD = 0.92

        # Game settings
        self.NUM_OBSTACLES = 20
        self.OBSTACLE_MIN_RADIUS = 15
        self.OBSTACLE_MAX_RADIUS = 30
        self.FINISH_LINE_Y = 50
        self.NEAR_MISS_RADIUS = 50

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24)
        
        # --- State Variables ---
        self.player_pos = None
        self.player_angle = None
        self.player_speed = None
        self.player_rect = None
        self.is_drifting = None
        self.obstacles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.time_left = None
        self.game_over = None
        self.win = None
        self.termination_reason = None
        self.rng = None

        # Initialize state
        self.reset()
        
        # --- Final Validation ---
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_angle = -math.pi / 2  # Start facing up
        self.player_speed = 0
        self.player_rect = pygame.Rect(0, 0, 18, 28)
        self.is_drifting = False
        
        self.steps = 0
        self.score = 0
        self.time_left = self.MAX_TIME
        self.game_over = False
        self.win = False
        self.termination_reason = ""

        self.particles = []
        self._generate_obstacles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, no-op still returns a new frame
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, _, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, shift_held)

        # --- Physics & State Update ---
        self._update_physics()
        self._update_particles()
        self.steps += 1
        self.time_left = max(0, self.MAX_TIME - (self.steps / self.FPS))
        
        # --- Termination Checks ---
        terminated = self._check_termination()
        self.game_over = terminated

        # --- Reward Calculation ---
        reward = self._calculate_reward(terminated)
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, movement, shift_held):
        self.is_drifting = shift_held and self.player_speed > self.MAX_SPEED * 0.5

        turn_mod = self.DRIFT_TURN_MOD if self.is_drifting else 1.0
        if movement == 3:  # Left
            self.player_angle -= self.TURN_SPEED * turn_mod
        if movement == 4:  # Right
            self.player_angle += self.TURN_SPEED * turn_mod

        if movement == 1:  # Up
            self.player_speed += self.ACCELERATION
        elif movement == 2:  # Down
            self.player_speed -= self.BRAKING
        
        if self.is_drifting:
            # Sound effect placeholder: # play_drift_sound()
            if self.steps % 3 == 0: # Create smoke particles while drifting
                self._create_drift_smoke()

    def _update_physics(self):
        # Apply friction
        friction = self.FRICTION * self.DRIFT_FRICTION_MOD if self.is_drifting else self.FRICTION
        self.player_speed *= friction
        self.player_speed = np.clip(self.player_speed, 0, self.MAX_SPEED)

        # Update position
        velocity = pygame.math.Vector2(math.cos(self.player_angle), math.sin(self.player_angle)) * self.player_speed
        self.player_pos += velocity

        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.HEIGHT)

        # Update player rect for collision and rendering
        self.player_rect.center = self.player_pos

    def _check_termination(self):
        # Time out
        if self.steps >= self.MAX_STEPS:
            self.termination_reason = "Time Out"
            return True

        # Finish line
        if self.player_rect.top < self.FINISH_LINE_Y:
            self.termination_reason = "Finished"
            self.win = True
            # Sound effect placeholder: # play_win_jingle()
            return True

        # Obstacle collision
        for pos, radius in self.obstacles:
            if self._check_rect_circle_collision(self.player_rect, pos, radius):
                self.termination_reason = "Collision"
                # Sound effect placeholder: # play_explosion_sound()
                self._create_explosion(self.player_pos)
                return True
        
        return False

    def _calculate_reward(self, terminated):
        if terminated:
            if self.termination_reason == "Collision":
                return -100.0
            if self.termination_reason == "Time Out":
                return -10.0
            if self.termination_reason == "Finished":
                time_bonus = 50 * (self.time_left / self.MAX_TIME)
                return 10.0 + time_bonus
        
        reward = 0.1  # Survival reward

        # Near miss penalty
        min_dist = float('inf')
        for pos, _ in self.obstacles:
            dist = self.player_pos.distance_to(pos)
            min_dist = min(min_dist, dist)
        
        if min_dist < self.NEAR_MISS_RADIUS:
            reward -= 5.0

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_left": round(self.time_left, 2)}

    def _render_game(self):
        # Finish line
        for i in range(0, self.WIDTH, 20):
            pygame.draw.rect(self.screen, self.COLOR_FINISH_LINE, (i, self.FINISH_LINE_Y - 5, 10, 5))
            pygame.draw.rect(self.screen, self.COLOR_FINISH_LINE, (i + 10, self.FINISH_LINE_Y, 10, 5))

        # Particles
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] > 0:
                alpha = int(255 * (p["life"] / p["max_life"]))
                if p["type"] == "line":
                     pygame.draw.aaline(self.screen, p["color"] + (alpha,), p["pos"], p["pos"] + p["vel"] * 3, 1)
                else: # circle
                    pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), p["color"] + (alpha,))

        # Obstacles
        for pos, radius in self.obstacles:
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), self.COLOR_OBSTACLE_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius * 0.9), self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), self.COLOR_OBSTACLE_GLOW)

        # Player
        if not (self.game_over and self.termination_reason == "Collision"):
            rotated_surf = pygame.transform.rotate(
                pygame.Surface(self.player_rect.size, pygame.SRCALPHA), math.degrees(-self.player_angle)
            )
            rotated_rect = rotated_surf.get_rect(center=self.player_rect.center)
            
            # Draw glow
            glow_rect = self.player_rect.inflate(8, 8)
            glow_surf = pygame.transform.rotate(
                pygame.Surface(glow_rect.size, pygame.SRCALPHA), math.degrees(-self.player_angle)
            )
            glow_surf_rect = glow_surf.get_rect(center=self.player_rect.center)
            pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=8)
            self.screen.blit(glow_surf, glow_surf_rect)

            # Draw main body
            pygame.draw.rect(rotated_surf, self.COLOR_PLAYER, rotated_surf.get_rect(), border_radius=4)
            self.screen.blit(rotated_surf, rotated_rect)

    def _render_ui(self):
        # Timer
        time_text = f"TIME: {self.time_left:.2f}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))
        
        # Game Over Message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(msg_surf, msg_rect)

            reason_surf = self.font_small.render(self.termination_reason, True, self.COLOR_UI_TEXT)
            reason_rect = reason_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 20))
            self.screen.blit(reason_surf, reason_rect)

    def _generate_obstacles(self):
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            while True:
                radius = self.rng.integers(self.OBSTACLE_MIN_RADIUS, self.OBSTACLE_MAX_RADIUS + 1)
                pos = pygame.math.Vector2(
                    self.rng.integers(radius, self.WIDTH - radius),
                    self.rng.integers(self.FINISH_LINE_Y + radius + 20, self.HEIGHT - 100) # Keep start and end clear
                )
                # Ensure no overlap with other obstacles
                if not any(pos.distance_to(obs_pos) < radius + obs_rad + 10 for obs_pos, obs_rad in self.obstacles):
                    self.obstacles.append((pos, radius))
                    break

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _create_drift_smoke(self):
        angle = self.player_angle + math.pi + self.rng.uniform(-0.2, 0.2)
        speed = 1.5
        pos_offset = pygame.math.Vector2(math.cos(self.player_angle + math.pi/2), math.sin(self.player_angle + math.pi/2)) * (10 if self.rng.random() > 0.5 else -10)
        self.particles.append({
            "pos": self.player_pos + pos_offset,
            "vel": pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed,
            "life": 15, "max_life": 15, "radius": self.rng.integers(2, 5),
            "color": self.COLOR_DRIFT_SMOKE, "type": "circle"
        })

    def _create_explosion(self, position):
        for _ in range(50):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 6)
            self.particles.append({
                "pos": pygame.math.Vector2(position),
                "vel": pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "life": 20, "max_life": 20,
                "color": self.COLOR_EXPLOSION, "type": "line"
            })

    def _check_rect_circle_collision(self, rect, circle_pos, circle_radius):
        # Find the closest point on the rect to the circle's center
        closest_x = np.clip(circle_pos.x, rect.left, rect.right)
        closest_y = np.clip(circle_pos.y, rect.top, rect.bottom)
        
        distance_x = circle_pos.x - closest_x
        distance_y = circle_pos.y - closest_y
        
        return (distance_x**2 + distance_y**2) < (circle_radius**2)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Arcade Racer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    total_steps = 0
    
    print("--- Arcade Racer ---")
    print(env.user_guide)

    while not done:
        # --- Player Input ---
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
        total_steps += 1
        
        # --- Rendering ---
        # Pygame uses a different coordinate system for surfaces, so we need to flip and rotate
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)

    print(f"\n--- Game Over ---")
    print(f"Termination Reason: {env.termination_reason}")
    print(f"Total Steps: {total_steps}")
    print(f"Final Score: {total_reward:.2f}")

    env.close()