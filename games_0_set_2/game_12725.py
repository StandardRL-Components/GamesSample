import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:28:17.632356
# Source Brief: brief_02725.md
# Brief Index: 2725
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a cannon to shoot falling centipede segments.
    The goal is to clear all segments before they reach the ground.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a cannon to shoot falling centipede segments. Clear all segments before they reach the ground."
    )
    user_guide = (
        "Use the ↑ and ↓ arrow keys to aim the cannon. Press space to fire."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_Y = 370
    MAX_STEPS = 2000
    FPS = 30

    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_GROUND = (60, 40, 30)
    COLOR_CANNON = (200, 200, 220)
    COLOR_CANNON_OUTLINE = (120, 120, 140)
    COLOR_PROJECTILE = (100, 255, 100)
    COLOR_PROJECTILE_CORE = (220, 255, 220)
    COLOR_CENTIPEDE = (255, 80, 80)
    COLOR_CENTIPEDE_CORE = (255, 180, 180)
    COLOR_EXPLOSION = (255, 200, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_GAMEOVER = (255, 50, 50)
    COLOR_WIN = (50, 255, 50)

    # Game parameters
    INITIAL_AMMO = 25
    INITIAL_CENTIPEDES = 8
    CANNON_ANGLE_SPEED = 1.0  # degrees per step
    CANNON_MIN_ANGLE = -85
    CANNON_MAX_ANGLE = 85
    PROJECTILE_SPEED = 15.0
    PROJECTILE_GRAVITY = 0.4
    MIN_SEGMENT_SIZE = 8
    SEGMENT_SPLIT_FACTOR = 0.75

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.cannon_angle = 0.0
        self.ammo = 0
        self.last_space_held = False
        self.fall_speed = 1.0

        self.projectiles = []
        self.centipedes = []
        self.particles = []


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.cannon_angle = 0.0
        self.ammo = self.INITIAL_AMMO
        self.last_space_held = False
        self.fall_speed = 1.0

        self.projectiles.clear()
        self.centipedes.clear()
        self.particles.clear()

        for _ in range(self.INITIAL_CENTIPEDES):
            self._spawn_centipede(
                pos=pygame.Vector2(
                    self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
                    self.np_random.uniform(-150, -50)
                ),
                size=self.np_random.uniform(20, 30)
            )

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        if not self.game_over:
            self._handle_input(action)
            reward += self._update_game_state()
            self._handle_collisions()
            reward += self._check_termination()

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            truncated = True
        
        terminated = self.game_over or truncated
        
        if terminated and self.steps >= self.MAX_STEPS and not self.game_over:
             # Timeout is a neutral outcome unless already won/lost
             reward = 0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        # Adjust cannon angle
        if movement == 1:  # Up
            self.cannon_angle -= self.CANNON_ANGLE_SPEED
        elif movement == 2:  # Down
            self.cannon_angle += self.CANNON_ANGLE_SPEED
        self.cannon_angle = np.clip(self.cannon_angle, self.CANNON_MIN_ANGLE, self.CANNON_MAX_ANGLE)

        # Fire projectile on button press (rising edge)
        if space_held and not self.last_space_held and self.ammo > 0:
            self._fire_projectile()
        self.last_space_held = space_held

    def _fire_projectile(self):
        # sfx: pew!
        self.ammo -= 1
        angle_rad = math.radians(self.cannon_angle - 90) # Adjust for pygame coordinates
        vel = pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * self.PROJECTILE_SPEED
        pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.GROUND_Y - 20)
        self.projectiles.append({"pos": pos, "vel": vel})

    def _update_game_state(self):
        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.fall_speed = min(5.0, self.fall_speed + 0.05)

        # Update projectiles
        for p in self.projectiles[:]:
            p["vel"].y += self.PROJECTILE_GRAVITY
            p["pos"] += p["vel"]
            if not self.screen.get_rect().collidepoint(p["pos"]):
                self.projectiles.remove(p)

        # Update centipedes
        downward_movement_reward = 0
        for c in self.centipedes:
            c["pos"].y += self.fall_speed
            downward_movement_reward += 0.1 * self.fall_speed

        # Update particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
        
        return downward_movement_reward

    def _handle_collisions(self):
        for proj in self.projectiles[:]:
            for cent in self.centipedes[:]:
                if proj["pos"].distance_to(cent["pos"]) < cent["size"]:
                    # sfx: boom!
                    self.projectiles.remove(proj)
                    self._create_explosion(cent["pos"], cent["size"])
                    self._split_centipede(cent)
                    self.centipedes.remove(cent)
                    self.score += 1
                    # Break inner loop since projectile is gone
                    break

    def _split_centipede(self, centipede):
        new_size = centipede["size"] * self.SEGMENT_SPLIT_FACTOR
        if new_size >= self.MIN_SEGMENT_SIZE:
            for _ in range(2):
                self._spawn_centipede(pos=centipede["pos"].copy(), size=new_size)

    def _spawn_centipede(self, pos, size):
        self.centipedes.append({"pos": pos, "size": size})

    def _create_explosion(self, pos, size):
        num_particles = int(size)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({"pos": pos.copy(), "vel": vel, "life": life, "max_life": life})

    def _check_termination(self):
        reward = 0
        # Loss condition: centipede reaches ground
        for c in self.centipedes:
            if c["pos"].y + c["size"] > self.GROUND_Y:
                self.game_over = True
                self.win = False
                reward = -100
                return reward
        
        # Win condition: all centipedes cleared
        if not self.centipedes:
            self.game_over = True
            self.win = True
            reward = 100
            return reward

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw ground
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 5)

        # Draw cannon
        cannon_len = 30
        cannon_width = 16
        angle_rad = math.radians(self.cannon_angle - 90)
        base_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.GROUND_Y - 10)
        
        p1 = base_pos + pygame.Vector2(-cannon_width / 2, 0).rotate(self.cannon_angle)
        p2 = base_pos + pygame.Vector2(cannon_width / 2, 0).rotate(self.cannon_angle)
        p3 = p2 + pygame.Vector2(0, -cannon_len).rotate(self.cannon_angle)
        p4 = p1 + pygame.Vector2(0, -cannon_len).rotate(self.cannon_angle)

        pygame.gfxdraw.aapolygon(self.screen, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)], self.COLOR_CANNON_OUTLINE)
        pygame.gfxdraw.filled_polygon(self.screen, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)], self.COLOR_CANNON)
        pygame.gfxdraw.filled_circle(self.screen, int(base_pos.x), int(base_pos.y), int(cannon_width/2 + 2), self.COLOR_CANNON)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*self.COLOR_EXPLOSION, alpha)
            radius = int(p["life"] / 5)
            # Use a temporary surface for alpha blending
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color)
            self.screen.blit(temp_surf, (int(p["pos"].x) - radius, int(p["pos"].y) - radius))


        # Draw centipedes
        for c in self.centipedes:
            x, y, size = int(c["pos"].x), int(c["pos"].y), int(c["size"])
            # Glow effect with alpha blending
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, size, size, size, (*self.COLOR_CENTIPEDE, 50))
            self.screen.blit(temp_surf, (x - size, y - size))

            pygame.gfxdraw.filled_circle(self.screen, x, y, int(size * 0.9), self.COLOR_CENTIPEDE)
            pygame.gfxdraw.filled_circle(self.screen, x, y, int(size * 0.6), self.COLOR_CENTIPEDE_CORE)

        # Draw projectiles
        for p in self.projectiles:
            x, y = int(p["pos"].x), int(p["pos"].y)
            # Glow effect with alpha blending
            radius = 8
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (*self.COLOR_PROJECTILE, 60))
            self.screen.blit(temp_surf, (x - radius, y - radius))

            pygame.gfxdraw.filled_circle(self.screen, x, y, 6, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 3, self.COLOR_PROJECTILE_CORE)

    def _render_ui(self):
        # Score and Ammo display
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        ammo_text = self.font_small.render(f"AMMO: {self.ammo}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(ammo_text, (self.SCREEN_WIDTH - ammo_text.get_width() - 10, 10))

        # Game Over / Win display
        if self.game_over:
            if self.win:
                end_text = self.font_large.render("YOU WIN!", True, self.COLOR_WIN)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_GAMEOVER)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "ammo": self.ammo}

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("Centipede Cascade")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)

    env.close()