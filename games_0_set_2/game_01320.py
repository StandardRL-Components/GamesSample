
# Generated: 2025-08-27T16:45:25.577179
# Source Brief: brief_01320.md
# Brief Index: 1320

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ←→ to adjust power, ↑↓ to adjust angle. Press Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down target practice game. Eliminate all targets with limited ammunition. Each shot counts!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRAVITY = 0.1
    MAX_AMMO = 10
    NUM_TARGETS = 25
    MIN_ANGLE, MAX_ANGLE = 5, 85  # degrees
    MIN_POWER, MAX_POWER = 2, 12
    ANGLE_STEP = 1.0
    POWER_STEP = 0.2

    # --- Colors ---
    COLOR_BG = (15, 18, 22)
    COLOR_LAUNCHER = (210, 210, 220)
    COLOR_AIM_GUIDE = (255, 255, 255, 60)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_TRAIL = (255, 200, 0)
    COLOR_TARGET_ALIVE = (220, 50, 50)
    COLOR_TARGET_HIT_FLASH = (100, 255, 100)
    COLOR_TARGET_DESTROYED = (80, 80, 80)
    COLOR_UI_TEXT = (200, 200, 220)
    COLOR_UI_VALUE = (255, 255, 255)
    COLOR_UI_SUCCESS = (100, 255, 100)
    COLOR_UI_FAIL = (255, 100, 100)
    COLOR_UI_INFO = (150, 180, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_info = pygame.font.SysFont("Consolas", 16)
        self.font_game_over = pygame.font.SysFont("Verdana", 48, bold=True)
        self.font_shot_result = pygame.font.SysFont("Verdana", 24, bold=True)

        self.launcher_pos = np.array([50.0, self.HEIGHT - 50.0])

        # Initialize state variables in reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.ammo = 0
        self.launch_angle = 0.0
        self.launch_power = 0.0
        self.projectile = None
        self.targets = []
        self.last_shot_result = ("", self.COLOR_UI_INFO)
        self.successful_hits = 0

        # Run validation
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ammo = self.MAX_AMMO
        self.launch_angle = 45.0
        self.launch_power = (self.MIN_POWER + self.MAX_POWER) / 2
        self.projectile = None
        self.last_shot_result = ("", self.COLOR_UI_INFO)
        self.successful_hits = 0

        self._generate_targets()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1

        if self.projectile:
            reward += self._update_projectile()
        else:
            self._handle_player_action(action)

        # Check for termination conditions
        all_targets_destroyed = all(not t['alive'] for t in self.targets)
        out_of_ammo = self.ammo == 0 and not self.projectile

        if all_targets_destroyed:
            terminated = True
            reward += 50  # Win bonus
            self.last_shot_result = ("VICTORY!", self.COLOR_UI_SUCCESS)
        elif out_of_ammo:
            terminated = True
            reward += -50  # Loss penalty
            self.last_shot_result = ("DEFEAT", self.COLOR_UI_FAIL)

        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _generate_targets(self):
        self.targets = []
        zone_width = (self.WIDTH - self.launcher_pos[0] - 100) / 5
        for i in range(self.NUM_TARGETS):
            tier = i // 5  # 0 to 4
            
            # Difficulty scaling
            radius = max(5, 15 - tier * 2)
            min_x = self.launcher_pos[0] + 50 + tier * zone_width
            max_x = min_x + zone_width
            
            pos_x = self.np_random.uniform(min_x, max_x)
            pos_y = self.np_random.uniform(50, self.HEIGHT - 50)
            
            self.targets.append({
                'pos': np.array([pos_x, pos_y]),
                'radius': radius,
                'base_radius': radius, # For reward calculation
                'alive': True,
                'hit_flash': 0
            })

    def _handle_player_action(self, action):
        movement, space_pressed, _ = action
        
        # Adjust aim
        if movement == 1:  # Up
            self.launch_angle = min(self.MAX_ANGLE, self.launch_angle + self.ANGLE_STEP)
        elif movement == 2:  # Down
            self.launch_angle = max(self.MIN_ANGLE, self.launch_angle - self.ANGLE_STEP)
        elif movement == 3:  # Left
            self.launch_power = max(self.MIN_POWER, self.launch_power - self.POWER_STEP)
        elif movement == 4:  # Right
            self.launch_power = min(self.MAX_POWER, self.launch_power + self.POWER_STEP)

        # Fire projectile
        if space_pressed and self.ammo > 0:
            self.ammo -= 1
            angle_rad = math.radians(self.launch_angle)
            vel = np.array([
                math.cos(angle_rad) * self.launch_power,
                -math.sin(angle_rad) * self.launch_power,
            ])
            self.projectile = {
                'pos': self.launcher_pos.copy(),
                'vel': vel,
                'trail': []
            }
            # sound: launch_sound()
            self.last_shot_result = ("FIRED!", self.COLOR_UI_INFO)

    def _update_projectile(self):
        if not self.projectile:
            return 0

        # Update trail
        self.projectile['trail'].append(self.projectile['pos'].copy())
        if len(self.projectile['trail']) > 20:
            self.projectile['trail'].pop(0)

        # Update position and velocity
        self.projectile['vel'][1] += self.GRAVITY
        self.projectile['pos'] += self.projectile['vel']

        # Check for collision with targets
        for target in self.targets:
            if target['alive']:
                dist = np.linalg.norm(self.projectile['pos'] - target['pos'])
                if dist < target['radius']:
                    target['alive'] = False
                    target['hit_flash'] = 10 # frames
                    self.score += 10
                    self.successful_hits += 1
                    self.projectile = None
                    # sound: target_explosion()
                    self.last_shot_result = ("HIT!", self.COLOR_UI_SUCCESS)
                    
                    # Reward for hitting a smaller target
                    bonus = 5 if target['base_radius'] < 10 else 0
                    return 1 + bonus

        # Check for out of bounds
        px, py = self.projectile['pos']
        if not (0 < px < self.WIDTH and 0 < py < self.HEIGHT * 2):
            self.projectile = None
            # sound: miss_sound()
            self.last_shot_result = ("MISS", self.COLOR_UI_FAIL)
            return -0.1

        return 0 # Projectile is still in flight

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
            "ammo": self.ammo,
            "hits": self.successful_hits
        }

    def _render_game(self):
        # Update and draw targets
        for target in self.targets:
            if target['hit_flash'] > 0:
                color = self.COLOR_TARGET_HIT_FLASH
                target['hit_flash'] -= 1
            elif target['alive']:
                color = self.COLOR_TARGET_ALIVE
            else:
                color = self.COLOR_TARGET_DESTROYED
            
            pos = (int(target['pos'][0]), int(target['pos'][1]))
            radius = int(target['radius'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

        # Draw launcher
        pygame.gfxdraw.filled_circle(self.screen, int(self.launcher_pos[0]), int(self.launcher_pos[1]), 8, self.COLOR_LAUNCHER)
        pygame.gfxdraw.aacircle(self.screen, int(self.launcher_pos[0]), int(self.launcher_pos[1]), 8, self.COLOR_LAUNCHER)

        # Draw aiming guide or projectile
        if self.projectile:
            self._render_projectile()
        else:
            self._render_aiming_guide()
            
    def _render_aiming_guide(self):
        angle_rad = math.radians(self.launch_angle)
        sim_pos = self.launcher_pos.copy()
        sim_vel = np.array([
            math.cos(angle_rad) * self.launch_power,
            -math.sin(angle_rad) * self.launch_power,
        ])
        
        points = []
        for _ in range(30):
            sim_vel[1] += self.GRAVITY
            sim_pos += sim_vel
            if _ % 2 == 0: # Draw every other point
                points.append((int(sim_pos[0]), int(sim_pos[1])))
        
        if len(points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_AIM_GUIDE, False, points)

    def _render_projectile(self):
        # Draw trail
        for i, pos in enumerate(self.projectile['trail']):
            alpha = int(255 * (i / len(self.projectile['trail'])))
            trail_color = self.COLOR_TRAIL + (alpha,)
            radius = int(3 * (i / len(self.projectile['trail'])))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, trail_color)

        # Draw projectile
        pos = (int(self.projectile['pos'][0]), int(self.projectile['pos'][1]))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_PROJECTILE)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, self.COLOR_PROJECTILE)


    def _render_ui(self):
        # --- Helper for rendering text ---
        def draw_text(text, font, color, pos, align="left"):
            surface = font.render(text, True, color)
            rect = surface.get_rect()
            if align == "left":
                rect.topleft = pos
            elif align == "right":
                rect.topright = pos
            elif align == "center":
                rect.center = pos
            self.screen.blit(surface, rect)

        # --- Draw UI elements ---
        # Score
        draw_text("SCORE", self.font_main, self.COLOR_UI_TEXT, (self.WIDTH - 10, 10), align="right")
        draw_text(f"{self.score:04d}", self.font_main, self.COLOR_UI_VALUE, (self.WIDTH - 10, 35), align="right")

        # Ammo
        draw_text("AMMO", self.font_main, self.COLOR_UI_TEXT, (10, 10))
        ammo_color = self.COLOR_UI_FAIL if self.ammo <= 3 else self.COLOR_UI_VALUE
        draw_text(f"{self.ammo:02d}", self.font_main, ammo_color, (10, 35))

        # Aiming info
        draw_text(f"Angle: {self.launch_angle:.1f}°", self.font_info, self.COLOR_UI_TEXT, (10, self.HEIGHT - 50))
        draw_text(f"Power: {self.launch_power:.1f}", self.font_info, self.COLOR_UI_TEXT, (10, self.HEIGHT - 30))

        # Shot result
        text, color = self.last_shot_result
        if text:
            draw_text(text, self.font_shot_result, color, (self.WIDTH / 2, 40), align="center")

        # Game Over message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            msg = "VICTORY!" if all(not t['alive'] for t in self.targets) else "GAME OVER"
            color = self.COLOR_UI_SUCCESS if msg == "VICTORY!" else self.COLOR_UI_FAIL
            draw_text(msg, self.font_game_over, color, (self.WIDTH / 2, self.HEIGHT / 2 - 20), align="center")
            draw_text("Press RESET to play again", self.font_main, self.COLOR_UI_TEXT, (self.WIDTH / 2, self.HEIGHT / 2 + 40), align="center")

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# --- Example usage to test the environment ---
if __name__ == "__main__":
    # Set dummy video driver for headless execution if needed
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    # For interactive play
    pygame.display.set_caption("Target Practice")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    running = True
    terminated = False
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Movement
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Space
            if keys[pygame.K_SPACE]:
                action[1] = 1
                
            # Shift (unused in this game)
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
        
        # Render the observation to the display
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2)) # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # If terminated, wait for a key press to reset
        if terminated:
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN:
                         if event.key == pygame.K_r:
                            env.reset()
                            terminated = False
                            wait_for_reset = False

        env.clock.tick(60) # Control the interactive loop speed

    env.close()