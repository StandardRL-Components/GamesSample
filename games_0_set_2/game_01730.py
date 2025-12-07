
# Generated: 2025-08-27T18:05:32.343523
# Source Brief: brief_01730.md
# Brief Index: 1730

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move the reticle. Press space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Arcade target practice. Eliminate all targets with your limited ammunition to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_TARGET = (255, 50, 50)
        self.COLOR_TARGET_OUTLINE = (200, 40, 40)
        self.COLOR_RETICLE = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HIT_EFFECT = (255, 255, 100)
        self.COLOR_SHOT_EFFECT = (200, 200, 255)

        # Game parameters
        self.RETICLE_SPEED = 10
        self.INITIAL_AMMO = 10
        self.TARGET_COUNT = 5
        self.TARGET_RADIUS = 15
        self.MAX_STEPS = 1000

        # Game state variables are initialized in reset()
        self.reticle_pos = None
        self.targets = None
        self.ammo = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.space_was_held = None
        self.effects = None
        
        # Initialize state variables for the first time
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Commented out for final submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.ammo = self.INITIAL_AMMO
        self.space_was_held = False
        self.effects = []
        
        self.reticle_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        
        # Generate targets, ensuring they don't overlap
        self.targets = []
        padding = self.TARGET_RADIUS + 10
        for _ in range(self.TARGET_COUNT):
            while True:
                pos = self.np_random.integers(
                    [padding, padding], 
                    [self.WIDTH - padding, self.HEIGHT - padding], 
                    size=2,
                    dtype=int
                ).astype(float)
                
                is_overlapping = False
                for t in self.targets:
                    dist = np.linalg.norm(pos - t['pos'])
                    if dist < (self.TARGET_RADIUS + t['radius']) * 1.5:
                        is_overlapping = True
                        break
                if not is_overlapping:
                    self.targets.append({
                        'pos': pos, 
                        'alive': True, 
                        'radius': self.TARGET_RADIUS
                    })
                    break

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        # shift_held = action[2] == 1 # Reserved

        # --- Game Logic ---
        self.steps += 1
        reward = -0.1 # Small penalty per step to encourage efficiency

        # 1. Handle Reticle Movement
        if movement == 1: # Up
            self.reticle_pos[1] -= self.RETICLE_SPEED
        elif movement == 2: # Down
            self.reticle_pos[1] += self.RETICLE_SPEED
        elif movement == 3: # Left
            self.reticle_pos[0] -= self.RETICLE_SPEED
        elif movement == 4: # Right
            self.reticle_pos[0] += self.RETICLE_SPEED
        
        # Clamp reticle to screen bounds
        self.reticle_pos[0] = np.clip(self.reticle_pos[0], 0, self.WIDTH)
        self.reticle_pos[1] = np.clip(self.reticle_pos[1], 0, self.HEIGHT)

        # 2. Handle Firing
        # Fire on the rising edge of the space button press
        if space_held and not self.space_was_held and self.ammo > 0:
            self.ammo -= 1
            # sfx: player_shoot.wav
            self._create_effect(self.reticle_pos, self.COLOR_SHOT_EFFECT, 20, 30)

            hit_a_target = False
            for target in self.targets:
                if target['alive']:
                    dist = np.linalg.norm(self.reticle_pos - target['pos'])
                    if dist <= target['radius']:
                        target['alive'] = False
                        hit_a_target = True
                        self.score += 1
                        reward += 10.0
                        # sfx: target_hit.wav
                        self._create_effect(target['pos'], self.COLOR_HIT_EFFECT, 40, 40)
                        break # Assume one shot hits one target
        
        self.space_was_held = space_held

        # 3. Update effects
        self._update_effects()

        # 4. Check Termination Conditions
        all_targets_destroyed = all(not t['alive'] for t in self.targets)
        out_of_ammo = self.ammo <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS

        terminated = all_targets_destroyed or out_of_ammo or max_steps_reached
        self.game_over = terminated

        if all_targets_destroyed:
            reward += 50.0 # Victory bonus
            # sfx: victory.wav
            
        if out_of_ammo and not all_targets_destroyed:
            # sfx: failure.wav
            pass

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _create_effect(self, pos, color, max_radius, duration):
        self.effects.append({
            'pos': np.copy(pos),
            'color': color,
            'max_radius': max_radius,
            'duration': duration,
            'life': duration
        })

    def _update_effects(self):
        for effect in self.effects:
            effect['life'] -= 1
        self.effects = [e for e in self.effects if e['life'] > 0]

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render targets
        for target in self.targets:
            if target['alive']:
                x, y = int(target['pos'][0]), int(target['pos'][1])
                r = int(target['radius'])
                pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_TARGET)
                pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_TARGET_OUTLINE)

        # Render effects
        for effect in self.effects:
            progress = (effect['duration'] - effect['life']) / effect['duration']
            current_radius = int(effect['max_radius'] * math.sin(progress * math.pi / 2)) # Ease out
            alpha = int(255 * (1 - progress))
            if alpha > 0 and current_radius > 0:
                x, y = int(effect['pos'][0]), int(effect['pos'][1])
                color = (*effect['color'], alpha)
                
                # Create a temporary surface for alpha blending
                temp_surf = pygame.Surface((current_radius*2, current_radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, current_radius, current_radius, current_radius, color)
                pygame.gfxdraw.aacircle(temp_surf, current_radius, current_radius, current_radius, color)
                self.screen.blit(temp_surf, (x - current_radius, y - current_radius))


        # Render reticle
        rx, ry = int(self.reticle_pos[0]), int(self.reticle_pos[1])
        length = 10
        gap = 3
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx - length - gap, ry), (rx - gap, ry), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx + gap, ry), (rx + length + gap, ry), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx, ry - length - gap), (rx, ry - gap), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx, ry + gap), (rx, ry + length + gap), 2)
        pygame.gfxdraw.pixel(self.screen, rx, ry, self.COLOR_RETICLE)


    def _render_ui(self):
        # Render Ammo
        ammo_text = self.font.render(f"AMMO: {self.ammo}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (10, 10))

        # Render Score
        score_text = self.font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Render Game Over message
        if self.game_over:
            all_targets_destroyed = all(not t['alive'] for t in self.targets)
            message = "VICTORY!" if all_targets_destroyed else "OUT OF AMMO"
            color = (100, 255, 100) if all_targets_destroyed else (255, 100, 100)
            
            end_text = self.font.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ammo": self.ammo,
            "targets_left": sum(1 for t in self.targets if t['alive'])
        }

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Target Practice Environment")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(env.game_description)
    print(env.user_guide)

    while not done:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        # Since auto_advance is False, we only step when there's an event or key press
        # For human play, we want smooth control, so we step every frame.
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, just need to display it.
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose.
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        clock.tick(30) # Limit to 30 FPS for human play

    print(f"Game Over! Final Info: {info}")
    env.close()