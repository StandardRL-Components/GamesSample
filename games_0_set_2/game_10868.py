import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:04:12.256666
# Source Brief: brief_00868.md
# Brief Index: 868
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple class for managing visual effect particles."""
    def __init__(self, pos, vel, radius, color, lifetime):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.initial_lifetime = lifetime

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, surface):
        progress = self.lifetime / self.initial_lifetime
        current_radius = int(self.radius * progress)
        if current_radius > 0:
            alpha = int(255 * progress)
            temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (current_radius, current_radius), current_radius)
            surface.blit(temp_surf, self.pos - pygame.Vector2(current_radius, current_radius), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Pilot a drone to collect numbered orbs against the clock. "
        "Activate stealth mode for higher rewards and chain same-numbered orbs for a combo bonus."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the drone. "
        "Press space to toggle stealth mode."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_STEPS = 120 * FPS
    WIN_SCORE = 1500

    # Colors
    COLOR_BG = (10, 15, 25)
    COLOR_DRONE = (255, 255, 255)
    COLOR_DRONE_GLOW = (150, 200, 255)
    COLOR_DRONE_STEALTH = (100, 150, 255)
    COLOR_DRONE_STEALTH_GLOW = (200, 100, 255)
    COLOR_TRAIL = (100, 200, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    
    ORB_COLORS = [
        (255, 80, 80),   # 1: Red
        (255, 160, 80),  # 2: Orange
        (255, 255, 80),  # 3: Yellow
        (160, 255, 80),  # 4: Chartreuse
        (80, 255, 160),  # 5: Spring Green
        (80, 255, 255),  # 6: Cyan
        (80, 160, 255),  # 7: Azure
        (160, 80, 255),  # 8: Violet
        (255, 80, 255),  # 9: Magenta
        (255, 80, 160)   # 10: Rose
    ]

    # Drone
    DRONE_RADIUS = 10
    DRONE_ACCELERATION = 0.5
    DRONE_FRICTION = 0.96

    # Orbs
    INITIAL_ORB_COUNT = 20
    ORB_RADIUS = 12

    # Rewards
    REWARD_COLLECT_NORMAL = 0.1
    REWARD_COLLECT_STEALTH = 0.2
    REWARD_COMBO_BONUS = 5.0
    REWARD_WIN = 100.0
    REWARD_LOSE = -10.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # Initialize state variables that are not reset
        self.drone_pos = pygame.Vector2(0, 0)
        self.drone_vel = pygame.Vector2(0, 0)
        self.is_stealth = False
        self.last_space_state = False
        self.orbs = []
        self.particles = []
        self.last_orb_value_collected = -1
        self.consecutive_collection_count = 0
        
        # The reset method is called here to ensure a valid initial state.
        # However, the initial observation is not returned from __init__.
        # A call to env.reset() is still required by the user.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        
        self.drone_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.drone_vel = pygame.Vector2(0, 0)
        
        self.is_stealth = False
        self.last_space_state = False
        
        self.orbs = []
        for _ in range(self.INITIAL_ORB_COUNT):
            self._spawn_orb()
            
        self.particles = []
        
        self.last_orb_value_collected = -1
        self.consecutive_collection_count = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.time_remaining -= 1
        
        self._handle_input(action)
        self._update_drone()
        self._update_particles()
        reward += self._check_collisions()
        
        terminated = False
        if self.score >= self.WIN_SCORE:
            terminated = True
            reward += self.REWARD_WIN
            # SFX: Win
        elif self.time_remaining <= 0:
            terminated = True
            reward += self.REWARD_LOSE
            # SFX: Lose
            
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        acc = pygame.Vector2(0, 0)
        if movement == 1: acc.y = -1
        if movement == 2: acc.y = 1
        if movement == 3: acc.x = -1
        if movement == 4: acc.x = 1
        
        if acc.length() > 0:
            acc.scale_to_length(self.DRONE_ACCELERATION)
        self.drone_vel += acc

        if space_held and not self.last_space_state:
            self.is_stealth = not self.is_stealth
            # SFX: Stealth On/Off
        self.last_space_state = space_held

    def _update_drone(self):
        self.drone_vel *= self.DRONE_FRICTION
        if self.drone_vel.length() < 0.1:
            self.drone_vel = pygame.Vector2(0, 0)
            
        self.drone_pos += self.drone_vel
        
        self.drone_pos.x = np.clip(self.drone_pos.x, self.DRONE_RADIUS, self.SCREEN_WIDTH - self.DRONE_RADIUS)
        self.drone_pos.y = np.clip(self.drone_pos.y, self.DRONE_RADIUS, self.SCREEN_HEIGHT - self.DRONE_RADIUS)
        
        trail_particles = 2 if self.is_stealth else 1
        for _ in range(trail_particles):
            if self.drone_vel.length() > 1:
                offset = self.drone_vel.normalize().rotate(random.uniform(-30, 30)) * -self.DRONE_RADIUS
                vel = self.drone_vel * 0.1 + pygame.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
                lifetime = 30 if self.is_stealth else 20
                radius = 4 if self.is_stealth else 3
                self.particles.append(Particle(self.drone_pos + offset, vel, radius, self.COLOR_TRAIL, lifetime))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _check_collisions(self):
        reward = 0
        collected_orbs = []
        for i, orb in enumerate(self.orbs):
            dist = self.drone_pos.distance_to(orb["pos"])
            if dist < self.DRONE_RADIUS + self.ORB_RADIUS:
                collected_orbs.append(i)
                
                # Score update
                score_gain = orb["value"] * 10
                self.score += score_gain

                # Reward update
                reward += self.REWARD_COLLECT_STEALTH if self.is_stealth else self.REWARD_COLLECT_NORMAL

                # Combo check
                if orb["value"] == self.last_orb_value_collected:
                    self.consecutive_collection_count += 1
                else:
                    self.last_orb_value_collected = orb["value"]
                    self.consecutive_collection_count = 1
                
                if self.consecutive_collection_count >= 5:
                    reward += self.REWARD_COMBO_BONUS
                    self.consecutive_collection_count = 0 # Reset after bonus
                    # SFX: Combo Bonus
                
                # SFX: Orb Collect
                self._create_collection_effect(orb["pos"], self.ORB_COLORS[orb["value"] - 1])

        if collected_orbs:
            self.orbs = [orb for i, orb in enumerate(self.orbs) if i not in collected_orbs]
            for _ in range(len(collected_orbs)):
                self._spawn_orb()
        
        return reward

    def _spawn_orb(self):
        pos = pygame.Vector2(
            self.np_random.uniform(self.ORB_RADIUS, self.SCREEN_WIDTH - self.ORB_RADIUS),
            self.np_random.uniform(self.ORB_RADIUS, self.SCREEN_HEIGHT - self.ORB_RADIUS)
        )
        value = self.np_random.integers(1, 11)
        self.orbs.append({"pos": pos, "value": value, "radius": self.ORB_RADIUS})

    def _create_collection_effect(self, pos, color):
        for _ in range(20):
            angle = random.uniform(0, 360)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(speed, 0).rotate(angle)
            lifetime = random.randint(15, 30)
            radius = random.uniform(1, 4)
            self.particles.append(Particle(pos, vel, radius, color, lifetime))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for p in self.particles:
            p.draw(self.screen)

        for orb in self.orbs:
            pos = (int(orb["pos"].x), int(orb["pos"].y))
            color = self.ORB_COLORS[orb["value"] - 1]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], orb["radius"], color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], orb["radius"], color)

        drone_pos_int = (int(self.drone_pos.x), int(self.drone_pos.y))
        drone_color = self.COLOR_DRONE_STEALTH if self.is_stealth else self.COLOR_DRONE
        glow_color = self.COLOR_DRONE_STEALTH_GLOW if self.is_stealth else self.COLOR_DRONE_GLOW
        
        # Glow effect
        for i in range(4):
            alpha = 60 - i * 15
            radius = self.DRONE_RADIUS + i * 3
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (*glow_color, alpha))
            self.screen.blit(temp_surf, (drone_pos_int[0] - radius, drone_pos_int[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.filled_circle(self.screen, drone_pos_int[0], drone_pos_int[1], self.DRONE_RADIUS, drone_color)
        pygame.gfxdraw.aacircle(self.screen, drone_pos_int[0], drone_pos_int[1], self.DRONE_RADIUS, drone_color)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_seconds = self.time_remaining / self.FPS
        time_text = self.font_ui.render(f"TIME: {time_seconds:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

        if self.is_stealth:
            stealth_text = self.font_ui.render("STEALTH", True, self.COLOR_DRONE_STEALTH_GLOW)
            self.screen.blit(stealth_text, (self.SCREEN_WIDTH / 2 - stealth_text.get_width() / 2, self.SCREEN_HEIGHT - 35))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "VICTORY!" if self.score >= self.WIN_SCORE else "TIME UP"
            color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            
            end_text = self.font_game_over.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "is_stealth": self.is_stealth,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        obs, _ = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    # env.validate_implementation() # Uncomment to run validation
    obs, info = env.reset()
    done = False
    
    # --- Manual Play ---
    # Create a display window
    pygame.display.set_caption("Drone Orb Collector")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    action = env.action_space.sample()
    action[0] = 0 # No movement initially
    action[1] = 0 # Space released
    action[2] = 0 # Shift released
    
    clock = pygame.time.Clock()
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: action[0] = 4
        
        # Space
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
        if done:
            print(f"Game Over. Final Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds before closing
            
    env.close()