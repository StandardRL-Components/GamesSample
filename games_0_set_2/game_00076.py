
# Generated: 2025-08-27T12:31:39.950513
# Source Brief: brief_00076.md
# Brief Index: 76

        
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
    """
    A procedurally generated target practice game where an agent aims and shoots at moving targets with limited ammo.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move the reticle. Press space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A procedurally generated target practice game where an agent aims and shoots at moving targets with limited ammo."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
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
        
        # Fonts and Colors
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (20, 40, 80)
        self.COLOR_TARGET = (255, 50, 50)
        self.COLOR_TARGET_DYING = (255, 150, 50)
        self.COLOR_RETICLE = (200, 255, 255)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_HIT_FLASH = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)

        # Game constants
        self.RETICLE_SPEED = 8
        self.TARGET_RADIUS = 15
        self.MAX_STEPS = 1000
        self.WIN_SCORE = 15
        self.INITIAL_AMMO = 30
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.ammo = 0
        self.game_over = False
        self.reticle_pos = None
        self.targets = []
        self.particles = []
        self.shot_effects = []
        self.target_spawn_timer = 0
        self.current_target_base_speed = 0
        self.space_pressed_last_frame = False
        self.win_condition_met = False
        self.loss_condition_met = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.ammo = self.INITIAL_AMMO
        self.game_over = False
        self.win_condition_met = False
        self.loss_condition_met = False
        
        self.reticle_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.targets = []
        self.particles = []
        self.shot_effects = []
        
        self.current_target_base_speed = 1.0
        self.target_spawn_timer = 0
        self.space_pressed_last_frame = False

        for _ in range(3):
            self._spawn_target()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            
            self._handle_input(movement, space_held)
            self._update_game_state()
            
            # Calculate reward for hits in this frame
            hits_this_frame = sum(1 for effect in self.shot_effects if effect['is_hit'])
            reward += hits_this_frame * 0.1

            self.steps += 1
            
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.win_condition_met:
                reward += 100 # Goal-oriented reward for winning
            elif self.loss_condition_met:
                reward -= 100 # Goal-oriented penalty for losing

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Move reticle
        if movement == 1: self.reticle_pos[1] -= self.RETICLE_SPEED
        elif movement == 2: self.reticle_pos[1] += self.RETICLE_SPEED
        elif movement == 3: self.reticle_pos[0] -= self.RETICLE_SPEED
        elif movement == 4: self.reticle_pos[0] += self.RETICLE_SPEED
        
        self.reticle_pos[0] = np.clip(self.reticle_pos[0], 0, self.WIDTH)
        self.reticle_pos[1] = np.clip(self.reticle_pos[1], 0, self.HEIGHT)

        # Fire projectile
        fire_request = space_held and not self.space_pressed_last_frame
        if fire_request and self.ammo > 0:
            self._fire_shot()
        self.space_pressed_last_frame = space_held

    def _fire_shot(self):
        self.ammo -= 1
        # SFX: Laser_Shoot.wav
        
        hit_target = None
        for target in self.targets:
            if target['state'] == 'alive':
                dist = np.linalg.norm(self.reticle_pos - target['pos'])
                if dist <= target['radius']:
                    hit_target = target
                    break
        
        if hit_target:
            # SFX: Explosion.wav
            hit_target['state'] = 'dying'
            self.score += 1
            self._create_particles(hit_target['pos'], self.COLOR_HIT_FLASH, 30)
            self.shot_effects.append({'pos': self.reticle_pos.copy(), 'timer': 10, 'is_hit': True})
        else:
            # SFX: Ricochet.wav
            self.shot_effects.append({'pos': self.reticle_pos.copy(), 'timer': 5, 'is_hit': False})

    def _update_game_state(self):
        # Update difficulty
        self.current_target_base_speed = 1.0 + (self.steps // 200) * 0.1
        
        # Update spawner
        self.target_spawn_timer -= 1
        if self.target_spawn_timer <= 0:
            self._spawn_target()
            spawn_interval = max(30, 90 - (self.steps // 100))
            self.target_spawn_timer = self.np_random.integers(spawn_interval, spawn_interval + 20)
            
        # Update targets
        new_targets = []
        for target in self.targets:
            if target['state'] == 'alive':
                target['pos'] += target['vel']
                if target['pos'][0] < -self.TARGET_RADIUS or target['pos'][0] > self.WIDTH + self.TARGET_RADIUS:
                    continue # Despawn if off-screen
                new_targets.append(target)
            elif target['state'] == 'dying':
                target['radius'] -= 1.5
                if target['radius'] > 0:
                    new_targets.append(target)
        self.targets = new_targets

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

        # Update shot effects
        self.shot_effects = [s for s in self.shot_effects if s['timer'] > 0]
        for s in self.shot_effects:
            s['timer'] -= 1

    def _spawn_target(self):
        side = self.np_random.choice(['left', 'right'])
        speed = self.np_random.uniform(self.current_target_base_speed, self.current_target_base_speed + 1.0)
        
        if side == 'left':
            pos = np.array([-self.TARGET_RADIUS, self.np_random.uniform(self.TARGET_RADIUS, self.HEIGHT - self.TARGET_RADIUS)], dtype=np.float32)
            vel = np.array([speed, 0], dtype=np.float32)
        else:
            pos = np.array([self.WIDTH + self.TARGET_RADIUS, self.np_random.uniform(self.TARGET_RADIUS, self.HEIGHT - self.TARGET_RADIUS)], dtype=np.float32)
            vel = np.array([-speed, 0], dtype=np.float32)
            
        self.targets.append({'pos': pos, 'vel': vel, 'radius': self.TARGET_RADIUS, 'state': 'alive'})

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'color': color})

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.win_condition_met = True
            return True
        if self.ammo <= 0 and all(t['state'] != 'alive' for t in self.targets) and not self.shot_effects:
            self.loss_condition_met = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.loss_condition_met = True # Treat timeout as a loss
            return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a gradient background
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, p['color'] + (alpha,))
        
        # Draw shot effects (tracer lines and impact flashes)
        for s in self.shot_effects:
            alpha = int(255 * (s['timer'] / 10))
            color = self.COLOR_PROJECTILE + (alpha,)
            start_pos = (self.WIDTH // 2, self.HEIGHT)
            end_pos = (int(s['pos'][0]), int(s['pos'][1]))
            pygame.draw.aaline(self.screen, color, start_pos, end_pos, 2)
            if s['is_hit']:
                pygame.gfxdraw.filled_circle(self.screen, end_pos[0], end_pos[1], int(15 * (1 - s['timer']/10)), self.COLOR_HIT_FLASH + (alpha,))
            else:
                pygame.gfxdraw.filled_circle(self.screen, end_pos[0], end_pos[1], 3, self.COLOR_PROJECTILE + (alpha,))

        # Draw targets
        for target in self.targets:
            pos_int = (int(target['pos'][0]), int(target['pos'][1]))
            radius = int(target['radius'])
            if radius > 0:
                color = self.COLOR_TARGET if target['state'] == 'alive' else self.COLOR_TARGET_DYING
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, color)

        # Draw reticle
        rx, ry = int(self.reticle_pos[0]), int(self.reticle_pos[1])
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx - 12, ry), (rx - 4, ry), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx + 4, ry), (rx + 12, ry), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx, ry - 12), (rx, ry - 4), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (rx, ry + 4), (rx, ry + 12), 2)
        pygame.gfxdraw.aacircle(self.screen, rx, ry, 8, self.COLOR_RETICLE)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos):
            shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_surface = font.render(text, True, color)
            self.screen.blit(shadow_surface, (pos[0] + 2, pos[1] + 2))
            self.screen.blit(text_surface, pos)

        # Draw Ammo
        ammo_text = f"AMMO: {self.ammo}"
        draw_text(ammo_text, self.font_main, self.COLOR_TEXT, (10, 10))
        
        # Draw Score
        score_text = f"HITS: {self.score}/{self.WIN_SCORE}"
        text_width = self.font_main.size(score_text)[0]
        draw_text(score_text, self.font_main, self.COLOR_TEXT, (self.WIDTH - text_width - 10, 10))

        # Draw Game Over message
        if self.game_over:
            if self.win_condition_met:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            text_w, text_h = self.font_game_over.size(msg)
            draw_text(msg, self.font_game_over, color, ((self.WIDTH - text_w) // 2, (self.HEIGHT - text_h) // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ammo": self.ammo,
            "game_over": self.game_over,
        }

    def close(self):
        pygame.font.quit()
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


if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # --- Action mapping for human play ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Score: {info['score']}, Steps: {info['steps']}")
            # Allow viewing the final screen for a moment
            final_frame = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(final_frame, (0, 0))
            pygame.display.flip()
            pygame.time.wait(3000)
            done = True

        # --- Render the observation to the screen ---
        frame = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame, (0, 0))
        pygame.display.flip()
        
        # --- Control the frame rate ---
        clock.tick(30) # Run at 30 FPS

    env.close()