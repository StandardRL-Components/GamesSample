import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:16:30.065441
# Source Brief: brief_02217.md
# Brief Index: 2217
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a glowing ship through a neon field. Collect orbs to gain speed, avoid obstacles, and reach the goal zone before time runs out."
    )
    user_guide = "Controls: Use the arrow keys (↑↓←→) to apply thrust to your ship."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1800  # 60 seconds at 30 FPS

    # Colors (Neon Inspired)
    COLOR_BG = (10, 10, 20)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_OBSTACLE = (255, 0, 80)
    COLOR_ORB = (0, 180, 255)
    COLOR_GOAL = (0, 255, 120)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SHOCKWAVE = (255, 255, 255)

    # Player settings
    PLAYER_RADIUS = 12
    PLAYER_FORCE = 1.0
    PLAYER_DRAG = 0.98

    # Game settings
    NUM_OBSTACLES = 5
    OBSTACLE_SIZE = (40, 40)
    NUM_ORBS = 10
    ORB_RADIUS = 8
    GOAL_WIDTH = 40

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_sub = pygame.font.SysFont("Consolas", 18)

        # --- State Variables ---
        self.player_pos = None
        self.player_vel = None
        self.speed_multiplier = None
        self.obstacles = []
        self.orbs = []
        self.particles = []
        self.animations = []
        self.steps = 0
        self.score = 0
        self.terminated = False

        # --- Initial Reset ---
        # Initialize state for validation, but proper reset is done by the user
        # self._initialize_state() is called in reset()

    def _initialize_state(self):
        """Initializes or resets all game state variables."""
        self.steps = 0
        self.score = 0
        self.terminated = False

        self.player_pos = np.array([80.0, self.SCREEN_HEIGHT / 2.0])
        self.player_vel = np.array([0.0, 0.0])
        self.speed_multiplier = 1.0

        self.obstacles = self._generate_obstacles()
        self.orbs = self._generate_orbs()

        self.particles = []
        self.animations = []

    def _generate_obstacles(self):
        obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            while True:
                x = self.np_random.uniform(self.SCREEN_WIDTH * 0.2, self.SCREEN_WIDTH * 0.8)
                y = self.np_random.uniform(0, self.SCREEN_HEIGHT - self.OBSTACLE_SIZE[1])
                rect = pygame.Rect(int(x), int(y), self.OBSTACLE_SIZE[0], self.OBSTACLE_SIZE[1])
                # Ensure no overlap with player start or goal
                if not rect.colliderect(pygame.Rect(self.player_pos[0]-50, self.player_pos[1]-50, 100, 100)) and \
                   not rect.colliderect(pygame.Rect(self.SCREEN_WIDTH - self.GOAL_WIDTH, 0, self.GOAL_WIDTH, self.SCREEN_HEIGHT)):
                    obstacles.append(rect)
                    break
        return obstacles

    def _generate_orbs(self):
        orbs = []
        for _ in range(self.NUM_ORBS):
            while True:
                pos = np.array([
                    self.np_random.uniform(self.ORB_RADIUS, self.SCREEN_WIDTH - self.ORB_RADIUS),
                    self.np_random.uniform(self.ORB_RADIUS, self.SCREEN_HEIGHT - self.ORB_RADIUS)
                ])
                # Ensure not inside an obstacle
                is_colliding = False
                for obs in self.obstacles:
                    if obs.collidepoint(pos[0], pos[1]):
                        is_colliding = True
                        break
                if not is_colliding:
                    orbs.append(pos)
                    break
        return orbs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        self._handle_input(action)
        self._update_physics()
        reward += self._handle_collisions()
        self._update_effects()

        self.terminated = self._check_termination()
        
        if self.terminated:
            if self.player_pos[0] >= self.SCREEN_WIDTH - self.GOAL_WIDTH - self.PLAYER_RADIUS:
                # Win condition
                reward += 100.0
                # sfx: win_sound
            elif any(self._collide_circle_rect(self.player_pos, self.PLAYER_RADIUS, obs) for obs in self.obstacles):
                # Loss by obstacle
                reward -= 5.0
                # sfx: player_death
        
        # Small penalty for existing to encourage speed
        reward -= 0.01

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.terminated = True

        return (
            self._get_observation(),
            reward,
            self.terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        # space_held and shift_held are ignored as per the brief
        # space_held = action[1] == 1
        # shift_held = action[2] == 1

        force = np.array([0.0, 0.0])
        if movement == 1:  # Up
            force[1] = -self.PLAYER_FORCE
        elif movement == 2:  # Down
            force[1] = self.PLAYER_FORCE
        elif movement == 3:  # Left
            force[0] = -self.PLAYER_FORCE
        elif movement == 4:  # Right
            force[0] = self.PLAYER_FORCE
        
        self.player_vel += force * self.speed_multiplier
        # sfx: thrust_sound (if movement > 0)

    def _update_physics(self):
        # Apply drag
        self.player_vel *= self.PLAYER_DRAG
        
        # Update position
        self.player_pos += self.player_vel

        # Wall bouncing
        if self.player_pos[0] < self.PLAYER_RADIUS:
            self.player_pos[0] = self.PLAYER_RADIUS
            self.player_vel[0] *= -0.8
            # sfx: bounce_wall
        if self.player_pos[0] > self.SCREEN_WIDTH - self.PLAYER_RADIUS:
            self.player_pos[0] = self.SCREEN_WIDTH - self.PLAYER_RADIUS
            self.player_vel[0] *= -0.8
            # sfx: bounce_wall
        if self.player_pos[1] < self.PLAYER_RADIUS:
            self.player_pos[1] = self.PLAYER_RADIUS
            self.player_vel[1] *= -0.8
            # sfx: bounce_wall
        if self.player_pos[1] > self.SCREEN_HEIGHT - self.PLAYER_RADIUS:
            self.player_pos[1] = self.SCREEN_HEIGHT - self.PLAYER_RADIUS
            self.player_vel[1] *= -0.8
            # sfx: bounce_wall

    def _handle_collisions(self):
        reward = 0
        
        # Orbs
        orbs_to_remove = []
        for i, orb_pos in enumerate(self.orbs):
            dist = np.linalg.norm(self.player_pos - orb_pos)
            if dist < self.PLAYER_RADIUS + self.ORB_RADIUS:
                orbs_to_remove.append(i)
                reward += 0.1  # Continuous feedback reward
                reward += 1.0  # Event-based "chain reaction" reward
                self.score += 1
                self.speed_multiplier *= 1.10
                # sfx: collect_orb
                # Trigger chain reaction animation
                self.animations.append({"type": "shockwave", "pos": orb_pos, "radius": 0, "max_radius": 60, "life": 15})

        if orbs_to_remove:
            self.orbs = [orb for i, orb in enumerate(self.orbs) if i not in orbs_to_remove]
        
        return reward

    def _update_effects(self):
        # Player trail particles
        if np.linalg.norm(self.player_vel) > 0.5:
            particle_pos = self.player_pos.copy() - self.player_vel * 2
            self.particles.append({
                "pos": particle_pos,
                "vel": self.np_random.uniform(-0.5, 0.5, size=2),
                "radius": self.np_random.uniform(2, 5),
                "life": 20,
                "color": self.COLOR_PLAYER
            })

        # Update and remove old particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            p["radius"] *= 0.95

        # Update and remove old animations
        self.animations = [a for a in self.animations if a["life"] > 0]
        for a in self.animations:
            a["life"] -= 1
            if a["type"] == "shockwave":
                a["radius"] += a["max_radius"] / 15

    def _check_termination(self):
        # Obstacle collision
        for obs in self.obstacles:
            if self._collide_circle_rect(self.player_pos, self.PLAYER_RADIUS, obs):
                return True
        
        # Goal reached
        if self.player_pos[0] >= self.SCREEN_WIDTH - self.GOAL_WIDTH - self.PLAYER_RADIUS:
            return True

        return False

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
            "speed_multiplier": self.speed_multiplier,
            "time_left_seconds": (self.MAX_STEPS - self.steps) / self.FPS
        }
    
    def _render_game(self):
        # Goal Zone
        goal_rect = pygame.Rect(self.SCREEN_WIDTH - self.GOAL_WIDTH, 0, self.GOAL_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, goal_rect, 0)
        
        # Animations (Shockwaves)
        for a in self.animations:
            if a["type"] == "shockwave":
                alpha = int(255 * (a['life'] / 15))
                color = (*self.COLOR_SHOCKWAVE, alpha)
                self._draw_glow_circle(
                    self.screen,
                    a["pos"].astype(int),
                    int(a["radius"]),
                    color,
                    num_layers=1,
                    glow_factor=1.0,
                    is_ring=True
                )

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*p["color"][:3], alpha)
            pos = p["pos"].astype(int)
            radius = max(0, int(p["radius"]))
            self._draw_glow_circle(self.screen, pos, radius, color, num_layers=2)

        # Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs, 0, border_radius=4)
        
        # Orbs
        for orb_pos in self.orbs:
            self._draw_glow_circle(self.screen, orb_pos.astype(int), self.ORB_RADIUS, self.COLOR_ORB)

        # Player
        self._draw_glow_circle(self.screen, self.player_pos.astype(int), self.PLAYER_RADIUS, self.COLOR_PLAYER, num_layers=5, glow_factor=2.0)
    
    def _render_ui(self):
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        text_surface = self.font_main.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 15, 10))

        # Speed Multiplier
        speed_text = f"SPEED: {self.speed_multiplier:.2f}x"
        text_surface = self.font_sub.render(speed_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 15, 40))

        # Score (Orbs Collected)
        score_text = f"ORBS: {int(self.score)}"
        text_surface = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (15, 10))

    def _draw_glow_circle(self, surface, pos, radius, color, num_layers=4, glow_factor=1.5, is_ring=False):
        base_color = color[:3]
        base_alpha = color[3] if len(color) > 3 else 255

        for i in range(num_layers, 0, -1):
            glow_radius = int(radius + (i - 1) * glow_factor)
            if glow_radius <= 0: continue
            
            alpha = int(base_alpha * (0.8 / num_layers) * (1 - (i / (num_layers * 2))))
            temp_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            
            if is_ring:
                pygame.draw.circle(temp_surface, (*base_color, alpha), (glow_radius, glow_radius), glow_radius, width=max(1, int(radius * 0.2)))
            else:
                 pygame.draw.circle(temp_surface, (*base_color, alpha), (glow_radius, glow_radius), glow_radius)

            surface.blit(temp_surface, (pos[0] - glow_radius, pos[1] - glow_radius))

        if not is_ring and radius > 0:
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, base_color)
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, base_color)

    def _collide_circle_rect(self, circle_pos, circle_radius, rect):
        closest_x = max(rect.left, min(circle_pos[0], rect.right))
        closest_y = max(rect.top, min(circle_pos[1], rect.bottom))
        
        distance_x = circle_pos[0] - closest_x
        distance_y = circle_pos[1] - closest_y
        
        distance_squared = (distance_x * distance_x) + (distance_y * distance_y)
        return distance_squared < (circle_radius * circle_radius)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Example Usage & Manual Play ---
    # The main loop needs a visible display, so we unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bouncing Ball Environment")
    
    obs, info = env.reset()
    done = False
    
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    running = True
    while running:
        movement_action = 0 # No-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                done = False

        keys = pygame.key.get_pressed()
        for key, action_val in key_map.items():
            if keys[key]:
                movement_action = action_val
                break 
        
        action = [movement_action, 0, 0]

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                print(f"Episode finished. Score: {info['score']}. Final Reward: {reward:.2f}")

        # The observation is what the agent sees. We need to convert it for display.
        # Pygame uses (width, height), numpy uses (height, width).
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    env.close()