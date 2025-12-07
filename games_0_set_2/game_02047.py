
# Generated: 2025-08-28T03:32:43.278932
# Source Brief: brief_02047.md
# Brief Index: 2047

        
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
        "Controls: ↑↓ to aim, ←→ to set power. Space to fire. Shift to reset aim."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Arcade target practice. Adjust your angle and power to hit all targets with limited ammo."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
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
        
        # Game constants
        self.GRAVITY = 0.15
        self.MAX_STEPS = 1000
        self.INITIAL_AMMO = 15
        self.NUM_TARGETS = 10
        self.CANNON_POS = (50, self.HEIGHT - 50)
        self.DEFAULT_ANGLE = -45.0
        self.DEFAULT_POWER = 50.0

        # Colors
        self.COLOR_BG_TOP = (15, 20, 45)
        self.COLOR_BG_BOTTOM = (30, 40, 80)
        self.COLOR_CANNON = (180, 180, 200)
        self.COLOR_PROJECTILE = (255, 80, 80)
        self.COLOR_TARGET_NORMAL = (0, 255, 150)
        self.COLOR_TARGET_SMALL = (0, 200, 255)
        self.COLOR_TARGET_OUTLINE = (255, 255, 255)
        self.COLOR_AIM_GUIDE = (255, 255, 255, 100)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_EXPLOSION = [(255, 255, 100), (255, 150, 50), (255, 50, 50)]

        # Fonts
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Verdana", 50, bold=True)

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ammo = 0
        self.angle = 0.0
        self.power = 0.0
        self.targets = []
        self.projectile = None
        self.particles = []
        self.np_random = None

        # Pre-render background for performance
        self.background = self._create_gradient_background()

        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ammo = self.INITIAL_AMMO
        self.angle = self.DEFAULT_ANGLE
        self.power = self.DEFAULT_POWER
        
        self.projectile = None
        self.particles = []
        
        self._generate_targets()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over:
            # If the game is over, do nothing and return the final state
            return (
                self._get_observation(), 0, True, False, self._get_info()
            )
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        shift_pressed = action[2] == 1  # Boolean
        
        # --- Update game logic based on action ---
        if shift_pressed:
            self.angle = self.DEFAULT_ANGLE
            self.power = self.DEFAULT_POWER
        
        if movement == 1:  # Up
            self.angle = max(-90, self.angle - 1.0)
        elif movement == 2:  # Down
            self.angle = min(0, self.angle + 1.0)
        elif movement == 3:  # Left
            self.power = max(10, self.power - 1.0)
        elif movement == 4:  # Right
            self.power = min(100, self.power + 1.0)

        if space_pressed and self.projectile is None and self.ammo > 0:
            self.ammo -= 1
            self._fire_projectile()
            # SFX: Cannon fire sound

        # --- Update physics ---
        # Update projectile
        if self.projectile:
            self.projectile['vel'][1] += self.GRAVITY
            self.projectile['pos'][0] += self.projectile['vel'][0]
            self.projectile['pos'][1] += self.projectile['vel'][1]
            self.projectile['trail'].append(tuple(self.projectile['pos']))
            if len(self.projectile['trail']) > 20:
                self.projectile['trail'].pop(0)

            # Check for target collision
            hit_target = False
            for i, target in enumerate(self.targets):
                if target['alive']:
                    dist = math.hypot(self.projectile['pos'][0] - target['pos'][0], self.projectile['pos'][1] - target['pos'][1])
                    if dist < target['radius'] + self.projectile['radius']:
                        target['alive'] = False
                        hit_target = True
                        current_reward = 1.0  # Base reward for hit
                        if target['type'] == 'small':
                            current_reward += 5.0  # Bonus for small target
                        self.score += int(current_reward)
                        reward += current_reward
                        self._create_explosion(target['pos'], target['radius'])
                        self.projectile = None
                        # SFX: Explosion sound
                        break
            
            # Check for projectile out of bounds (miss)
            if self.projectile and not hit_target:
                if (self.projectile['pos'][0] > self.WIDTH or 
                    self.projectile['pos'][1] > self.HEIGHT or
                    self.projectile['pos'][0] < 0):
                    reward -= 0.1 # Penalty for missing
                    self.projectile = None

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

        # --- Check for termination ---
        self.steps += 1
        terminated = False
        
        targets_remaining = sum(1 for t in self.targets if t['alive'])
        
        if targets_remaining == 0:
            # Win condition
            reward += 50.0
            self.score += 50
            terminated = True
            self.game_over = True
        elif self.ammo == 0 and self.projectile is None:
            # Lose condition: out of ammo
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            # Lose condition: out of time
            terminated = True
            self.game_over = True
            
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Draw background
        self.screen.blit(self.background, (0, 0))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw cannon
        angle_rad = math.radians(self.angle)
        barrel_length = 40
        end_x = self.CANNON_POS[0] + barrel_length * math.cos(angle_rad)
        end_y = self.CANNON_POS[1] + barrel_length * math.sin(angle_rad)
        pygame.draw.line(self.screen, self.COLOR_CANNON, self.CANNON_POS, (int(end_x), int(end_y)), 8)
        pygame.gfxdraw.filled_circle(self.screen, int(self.CANNON_POS[0]), int(self.CANNON_POS[1]), 15, self.COLOR_CANNON)
        pygame.gfxdraw.aacircle(self.screen, int(self.CANNON_POS[0]), int(self.CANNON_POS[1]), 15, self.COLOR_CANNON)

        # Draw aiming guide if not firing
        if self.projectile is None:
            self._draw_aiming_arc()

        # Draw targets
        for target in self.targets:
            if target['alive']:
                color = self.COLOR_TARGET_SMALL if target['type'] == 'small' else self.COLOR_TARGET_NORMAL
                pos = (int(target['pos'][0]), int(target['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], target['radius'], color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], target['radius'], self.COLOR_TARGET_OUTLINE)

        # Draw projectile and its trail
        if self.projectile:
            # Trail
            for i, p in enumerate(self.projectile['trail']):
                alpha = int(255 * (i / len(self.projectile['trail'])))
                color = (*self.COLOR_PROJECTILE, alpha)
                temp_surf = pygame.Surface((self.projectile['radius'], self.projectile['radius']), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (self.projectile['radius']//2, self.projectile['radius']//2), self.projectile['radius']//2)
                self.screen.blit(temp_surf, (int(p[0])-self.projectile['radius']//2, int(p[1])-self.projectile['radius']//2))

            # Projectile
            pos = (int(self.projectile['pos'][0]), int(self.projectile['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.projectile['radius'], self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.projectile['radius'], (255, 255, 255))
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, int(255 * (p['life'] / p['max_life'])))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

    def _render_ui(self):
        # Draw text info
        self._draw_text(f"SCORE: {self.score}", (10, 10), self.font_main, self.COLOR_UI_TEXT)
        self._draw_text(f"AMMO: {self.ammo}", (self.WIDTH - 120, 10), self.font_main, self.COLOR_UI_TEXT)
        
        # Draw power and angle indicators
        self._draw_text(f"Angle: {int(-self.angle):>2}°", (10, self.HEIGHT - 30), self.font_main, self.COLOR_UI_TEXT)
        self._draw_text(f"Power: {int(self.power):>3}", (150, self.HEIGHT - 30), self.font_main, self.COLOR_UI_TEXT)
        
        # Game over message
        if self.game_over:
            targets_remaining = sum(1 for t in self.targets if t['alive'])
            message = "YOU WIN!" if targets_remaining == 0 else "GAME OVER"
            color = self.COLOR_TARGET_NORMAL if targets_remaining == 0 else self.COLOR_PROJECTILE
            
            text_surf = self.font_game_over.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0,0))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ammo": self.ammo,
            "targets_remaining": sum(1 for t in self.targets if t['alive']),
        }

    def _generate_targets(self):
        self.targets = []
        for i in range(self.NUM_TARGETS):
            is_small = self.np_random.random() < 0.3 # 30% chance of being a small target
            radius = self.np_random.integers(10, 16) if is_small else self.np_random.integers(20, 31)
            x = self.np_random.integers(self.WIDTH // 2, self.WIDTH - radius - 20)
            y = self.np_random.integers(radius + 20, self.HEIGHT - radius - 20)
            
            # Simple check to avoid major overlaps
            is_overlapping = False
            for t in self.targets:
                dist = math.hypot(x - t['pos'][0], y - t['pos'][1])
                if dist < radius + t['radius'] + 10: # Add padding
                    is_overlapping = True
                    break
            
            if is_overlapping and i > 0:
                continue

            self.targets.append({
                'pos': [x, y],
                'radius': radius,
                'alive': True,
                'type': 'small' if is_small else 'normal'
            })
        
        # Ensure we have the correct number of targets
        while len(self.targets) < self.NUM_TARGETS:
             self._generate_targets() # Recurse if we failed to place enough

    def _fire_projectile(self):
        angle_rad = math.radians(self.angle)
        power_mult = self.power / 50.0 
        speed = 8.0 * power_mult
        
        vel_x = speed * math.cos(angle_rad)
        vel_y = speed * math.sin(angle_rad)
        
        self.projectile = {
            'pos': list(self.CANNON_POS),
            'vel': [vel_x, vel_y],
            'radius': 6,
            'trail': []
        }

    def _create_explosion(self, pos, radius):
        num_particles = 20 + int(radius)
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': list(pos),
                'vel': [speed * math.cos(angle), speed * math.sin(angle)],
                'life': life,
                'max_life': life,
                'radius': self.np_random.integers(2, 5),
                'color': random.choice(self.COLOR_EXPLOSION)
            })

    def _draw_aiming_arc(self):
        angle_rad = math.radians(self.angle)
        power_mult = self.power / 50.0 
        speed = 8.0 * power_mult
        
        pos = list(self.CANNON_POS)
        vel = [speed * math.cos(angle_rad), speed * math.sin(angle_rad)]
        
        for i in range(30): # Simulate 30 steps into the future
            vel[1] += self.GRAVITY
            pos[0] += vel[0]
            pos[1] += vel[1]
            if i % 3 == 0: # Draw a dot every 3 steps
                if 0 < pos[0] < self.WIDTH and 0 < pos[1] < self.HEIGHT:
                    temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, self.COLOR_AIM_GUIDE, (2, 2), 2)
                    self.screen.blit(temp_surf, (int(pos[0])-2, int(pos[1])-2))

    def _draw_text(self, text, pos, font, color):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _create_gradient_background(self):
        bg = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            # Linear interpolation
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / self.HEIGHT
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / self.HEIGHT
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / self.HEIGHT
            pygame.draw.line(bg, (int(r), int(g), int(b)), (0, y), (self.WIDTH, y))
        return bg

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example of how to run the environment for manual play
if __name__ == '__main__':
    env = GameEnv()
    
    # This part is for human testing and requires a visible pygame window.
    try:
        import os
        # Set a video driver that works for your system
        # e.g., "x11", "directfb", "fbcon" on Linux; "windib" on Windows
        if os.name == "posix":
            os.environ.setdefault("SDL_VIDEODRIVER", "x11")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption(env.game_description)
    except pygame.error:
        print("Could not create display. Manual play is disabled.")
        screen = None

    if screen:
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            # Convert observation back to a surface for display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # Action mapping for keyboard
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # Process quit events
            should_quit = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_quit = True
            if should_quit:
                break
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)

            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
            
            env.clock.tick(30)

        print(f"Final Score: {info['score']}")
    env.close()