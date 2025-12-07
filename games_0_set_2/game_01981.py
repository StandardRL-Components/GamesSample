
# Generated: 2025-08-28T03:17:49.730421
# Source Brief: brief_01981.md
# Brief Index: 1981

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to aim your jump. Press space to leap. Use left/right for air control."
    )

    # User-facing description of the game
    game_description = (
        "Leap between oscillating platforms to reach the goal. Time your jumps carefully and use air "
        "control to navigate. Reach the golden platform at the top as fast as you can for a high score!"
    )

    # Frames auto-advance for smooth, real-time physics
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRAVITY = 0.4
        self.JUMP_POWER_VERTICAL = 11
        self.JUMP_POWER_HORIZONTAL = 9
        self.AIR_CONTROL = 0.35
        self.PLAYER_SIZE = 16
        self.PLATFORM_HEIGHT = 12
        self.MAX_STEPS = 5000

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 10)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLATFORM = (150, 150, 170)
        self.COLOR_GOAL_PLATFORM = (255, 215, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_AIMER = (255, 100, 100)
        self.COLOR_PARTICLE = (200, 200, 255)
        
        # EXACT spaces
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.player_pos = [0, 0]
        self.player_vel = [0, 0]
        self.platforms = []
        self.particles = []
        self.aim_vector = [0, -1]
        self.current_platform_idx = -1
        self.on_platform = False
        self.is_airborne = True
        self.highest_y_pos = self.HEIGHT
        self.oscillation_speed = 1.0

        # This will call reset and initialize the state properly
        self.reset()

        # Run validation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # World state
        self.oscillation_speed = 1.0
        self._generate_platforms()
        
        # Player state
        start_platform = self.platforms[0]
        self.player_pos = [start_platform["rect"].centerx, start_platform["rect"].top - self.PLAYER_SIZE / 2]
        self.player_vel = [0, 0]
        self.on_platform = True
        self.is_airborne = False
        self.current_platform_idx = 0
        self.highest_y_pos = self.player_pos[1]
        
        # Aiming state
        self.aim_vector = [0, -1] # Default aim is straight up
        
        # Effects
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def _generate_platforms(self):
        self.platforms = []
        # Start platform
        start_plat_rect = pygame.Rect(self.WIDTH/2 - 50, self.HEIGHT - 40, 100, self.PLATFORM_HEIGHT)
        self.platforms.append({
            "rect": start_plat_rect,
            "base_y": start_plat_rect.y,
            "osc_phase": 0,
            "is_goal": False
        })
        
        # Procedural platforms
        current_y = self.HEIGHT - 130
        max_jump_height = (self.JUMP_POWER_VERTICAL**2) / (2 * self.GRAVITY) - 40 # Safety margin

        while current_y > 60:
            plat_width = self.np_random.integers(60, 120)
            plat_x = self.np_random.integers(0, self.WIDTH - plat_width)
            
            plat_rect = pygame.Rect(plat_x, int(current_y), plat_width, self.PLATFORM_HEIGHT)
            self.platforms.append({
                "rect": plat_rect,
                "base_y": plat_rect.y,
                "osc_phase": self.np_random.uniform(0, 2 * math.pi),
                "is_goal": False
            })
            
            y_gap = self.np_random.uniform(max_jump_height * 0.5, max_jump_height * 0.9)
            current_y -= y_gap
            
        # Goal platform
        goal_width = 150
        goal_x = self.WIDTH / 2 - goal_width / 2
        goal_rect = pygame.Rect(goal_x, 30, goal_width, self.PLATFORM_HEIGHT + 10)
        self.platforms.append({
            "rect": goal_rect,
            "base_y": goal_rect.y,
            "osc_phase": 0,
            "is_goal": True
        })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Time penalty
        
        # 1. Handle Player Input & Actions
        if self.on_platform:
            aim_map = {1: [0, -1], 2: [0, -0.5], 3: [-0.707, -0.707], 4: [0.707, -0.707]}
            if movement in aim_map:
                self.aim_vector = aim_map[movement]

            if space_pressed:
                # Sound effect placeholder: Jump
                self.player_vel[0] = self.aim_vector[0] * self.JUMP_POWER_HORIZONTAL
                self.player_vel[1] = self.aim_vector[1] * self.JUMP_POWER_VERTICAL
                self.on_platform = False
                self.is_airborne = True
                self.current_platform_idx = -1
        
        # 2. Update Physics & Game State
        self.steps += 1
        
        if self.is_airborne:
            if movement == 3: self.player_vel[0] -= self.AIR_CONTROL
            if movement == 4: self.player_vel[0] += self.AIR_CONTROL
            self.player_vel[1] += self.GRAVITY
            self.player_vel[0] *= 0.98
            self.player_pos[0] += self.player_vel[0]
            self.player_pos[1] += self.player_vel[1]

        if self.steps > 0 and self.steps % 500 == 0:
            self.oscillation_speed = min(3.0, self.oscillation_speed + 0.05)

        for i, plat in enumerate(self.platforms):
            if not plat["is_goal"] and i > 0:
                oscillation = math.sin(plat["osc_phase"] + self.steps * 0.05 * self.oscillation_speed) * 15
                plat["rect"].y = plat["base_y"] + oscillation

        # 3. Collision Detection
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE / 2, self.player_pos[1] - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        terminated = False
        if self.is_airborne and self.player_vel[1] > 0:
            for i, plat in enumerate(self.platforms):
                if player_rect.colliderect(plat["rect"]) and (player_rect.bottom - self.player_vel[1]) <= plat["rect"].top:
                    # Sound effect placeholder: Land
                    self.on_platform, self.is_airborne = True, False
                    self.player_vel = [0, 0]
                    self.player_pos[1] = plat["rect"].top - self.PLAYER_SIZE / 2
                    self.current_platform_idx = i
                    reward += 5.0
                    self._create_particles(player_rect.midbottom, 15)
                    
                    if plat["is_goal"]:
                        # Sound effect placeholder: Win
                        terminated = True
                        reward += 100.0
                        self._create_particles(player_rect.center, 50, self.COLOR_GOAL_PLATFORM)
                    break

        # 4. Update Rewards & Check Termination
        if self.player_pos[1] < self.highest_y_pos:
            reward += 0.1 * (self.highest_y_pos - self.player_pos[1])
            self.highest_y_pos = self.player_pos[1]
            
        if self.player_pos[1] > self.HEIGHT + self.PLAYER_SIZE:
            # Sound effect placeholder: Fall
            terminated = True
            reward += -50.0
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        if terminated: self.game_over = True
            
        # 5. Update Effects
        self._update_particles()
        
        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, pos, count, color=None):
        if color is None: color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos), "vel": vel, "color": color,
                "lifespan": self.np_random.integers(20, 40),
                "radius": self.np_random.uniform(1, 3.5)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # Particle gravity
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = [int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(self.COLOR_BG_TOP, self.COLOR_BG_BOTTOM)]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"] * (p["lifespan"] / 40)), p["color"])

        # Draw platforms
        for plat in self.platforms:
            color = self.COLOR_GOAL_PLATFORM if plat["is_goal"] else self.COLOR_PLATFORM
            shadow_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(self.screen, shadow_color, plat["rect"].move(0, 4), border_radius=3)
            pygame.draw.rect(self.screen, color, plat["rect"], border_radius=3)

        # Draw player and aimer
        player_rect = pygame.Rect(int(self.player_pos[0] - self.PLAYER_SIZE / 2), int(self.player_pos[1] - self.PLAYER_SIZE / 2), self.PLAYER_SIZE, self.PLAYER_SIZE)
        pygame.draw.rect(self.screen, (0,0,0), player_rect.inflate(4,4), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)

        if self.on_platform:
            aim_end_pos = (self.player_pos[0] + self.aim_vector[0] * 25, self.player_pos[1] + self.aim_vector[1] * 25)
            pygame.draw.line(self.screen, self.COLOR_AIMER, (int(self.player_pos[0]), int(self.player_pos[1])), (int(aim_end_pos[0]), int(aim_end_pos[1])), 2)
            pygame.gfxdraw.filled_circle(self.screen, int(aim_end_pos[0]), int(aim_end_pos[1]), 4, self.COLOR_AIMER)
            pygame.gfxdraw.aacircle(self.screen, int(aim_end_pos[0]), int(aim_end_pos[1]), 4, self.COLOR_AIMER)
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        height_val = max(0, int(self.platforms[0]["rect"].top - self.player_pos[1]))
        texts = [
            f"Height: {height_val}",
            f"Score: {self.score:.0f}",
            f"Time: {self.steps}"
        ]
        for i, text_str in enumerate(texts):
            text_surf = self.font_small.render(text_str, True, self.COLOR_TEXT)
            self.screen.blit(text_surf, (10, 10 + i * 20))

        if self.game_over:
            won = self.current_platform_idx != -1 and self.platforms[self.current_platform_idx]["is_goal"]
            end_text_str = "SUCCESS!" if won else "GAME OVER"
            end_color = self.COLOR_GOAL_PLATFORM if won else (200, 50, 50)
            
            end_text = self.font_large.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            
            shadow = self.font_large.render(end_text_str, True, (0,0,0))
            self.screen.blit(shadow, text_rect.move(2, 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": max(0, int(self.platforms[0]["rect"].top - self.player_pos[1])),
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Set SDL_VIDEODRIVER to a dummy value for headless execution
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Arcade Hopper")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # --- Game Loop ---
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # movement=none, space=released, shift=released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # The game will show the 'Game Over' screen for a few seconds before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
    env.close()