
# Generated: 2025-08-28T05:03:25.514992
# Source Brief: brief_02477.md
# Brief Index: 2477

        
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

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move the reticle. Press space to fire."
    )

    # User-facing game description
    game_description = (
        "A top-down target practice game. Eliminate all moving targets with limited ammunition. "
        "Precision and timing are key to achieving a high score."
    )

    # Frames auto-advance at 30fps
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1500  # 50 seconds at 30fps

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 44, 52)
    COLOR_TARGET = (224, 108, 117)
    COLOR_RETICLE = (152, 195, 121)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_TEXT = (192, 199, 209)
    COLOR_EXPLOSION = (229, 192, 123)
    
    # Game parameters
    NUM_TARGETS = 20
    INITIAL_AMMO = 40
    RETICLE_SPEED = 10
    TARGET_RADIUS = 15
    BASE_TARGET_SPEED = 1.5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reticle_pos = None
        self.ammo = 0
        self.targets_destroyed = 0
        self.current_target_speed = 0
        self.targets = []
        self.particles = []
        self.last_space_held = False
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reticle_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.ammo = self.INITIAL_AMMO
        self.targets_destroyed = 0
        self.current_target_speed = self.BASE_TARGET_SPEED
        self.targets = []
        self.particles = []
        self.last_space_held = False

        self._spawn_targets()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, _ = action
        reward = 0
        terminated = False

        if not self.game_over:
            # --- Handle Input ---
            self._move_reticle(movement)
            shot_fired = space_held and not self.last_space_held
            
            if shot_fired and self.ammo > 0:
                self.ammo -= 1
                # sfx: shoot
                self._create_muzzle_flash()
                hit_target = self._process_shot()
                
                if hit_target:
                    reward += 1.0
                    # sfx: explosion
                else:
                    reward -= 0.01
                    # sfx: miss
            
            self.last_space_held = bool(space_held)

            # --- Update Game State ---
            self._update_targets()
            self._update_particles()

            # --- Check Termination Conditions ---
            if self.targets_destroyed >= self.NUM_TARGETS:
                terminated = True
                self.game_over = True
                reward += 10.0  # Win bonus
            elif self.ammo <= 0 and not any(p['type'] == 'shot' for p in self.particles):
                terminated = True
                self.game_over = True
                reward -= 10.0  # Lose penalty
            elif self.steps >= self.MAX_STEPS:
                terminated = True
                self.game_over = True

        self.steps += 1
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_targets(self):
        self.targets = []
        for _ in range(self.NUM_TARGETS):
            edge = self.np_random.integers(0, 4)
            if edge == 0:  # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -self.TARGET_RADIUS)
                vel = pygame.Vector2(self.np_random.uniform(-0.5, 0.5), 1)
            elif edge == 1:  # Right
                pos = pygame.Vector2(self.SCREEN_WIDTH + self.TARGET_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT))
                vel = pygame.Vector2(-1, self.np_random.uniform(-0.5, 0.5))
            elif edge == 2:  # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.TARGET_RADIUS)
                vel = pygame.Vector2(self.np_random.uniform(-0.5, 0.5), -1)
            else:  # Left
                pos = pygame.Vector2(-self.TARGET_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT))
                vel = pygame.Vector2(1, self.np_random.uniform(-0.5, 0.5))
            
            vel.normalize_ip()
            self.targets.append({'pos': pos, 'vel': vel, 'alive': True})

    def _move_reticle(self, movement):
        if movement == 1:  # Up
            self.reticle_pos.y -= self.RETICLE_SPEED
        elif movement == 2:  # Down
            self.reticle_pos.y += self.RETICLE_SPEED
        elif movement == 3:  # Left
            self.reticle_pos.x -= self.RETICLE_SPEED
        elif movement == 4:  # Right
            self.reticle_pos.x += self.RETICLE_SPEED
        
        self.reticle_pos.x = np.clip(self.reticle_pos.x, 0, self.SCREEN_WIDTH)
        self.reticle_pos.y = np.clip(self.reticle_pos.y, 0, self.SCREEN_HEIGHT)

    def _process_shot(self):
        for target in self.targets:
            if target['alive']:
                dist = self.reticle_pos.distance_to(target['pos'])
                if dist <= self.TARGET_RADIUS:
                    target['alive'] = False
                    self.targets_destroyed += 1
                    self._create_explosion(target['pos'])

                    # Increase difficulty
                    if self.targets_destroyed > 0 and self.targets_destroyed % 5 == 0:
                        self.current_target_speed += 0.2

                    return True
        return False

    def _update_targets(self):
        for target in self.targets:
            if target['alive']:
                target['pos'] += target['vel'] * self.current_target_speed
                
                # Bounce off walls
                if target['pos'].x < self.TARGET_RADIUS or target['pos'].x > self.SCREEN_WIDTH - self.TARGET_RADIUS:
                    target['vel'].x *= -1
                if target['pos'].y < self.TARGET_RADIUS or target['pos'].y > self.SCREEN_HEIGHT - self.TARGET_RADIUS:
                    target['vel'].y *= -1

    def _create_muzzle_flash(self):
        # Shot originates from bottom center
        origin = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT)
        self.particles.append({
            'type': 'flash', 'pos': origin, 'radius': 20, 'lifespan': 3
        })
        
        # Create a tracer line particle
        self.particles.append({
            'type': 'tracer', 'start': origin, 'end': self.reticle_pos.copy(), 'lifespan': 5, 'max_lifespan': 5
        })

    def _create_explosion(self, pos):
        for _ in range(30):
            vel = pygame.Vector2(self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1))
            if vel.length() > 0:
                vel.normalize_ip()
            vel *= self.np_random.uniform(2, 6)
            self.particles.append({
                'type': 'explosion', 'pos': pos.copy(), 'vel': vel,
                'radius': self.np_random.uniform(2, 5), 'lifespan': self.np_random.integers(15, 30)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
                continue
            
            if p['type'] == 'explosion':
                p['pos'] += p['vel']
                p['radius'] *= 0.95  # Shrink
                p['vel'] *= 0.9  # Slow down
            elif p['type'] == 'flash':
                p['radius'] *= 0.75

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Draw targets
        for target in self.targets:
            if target['alive']:
                pos = (int(target['pos'].x), int(target['pos'].y))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.TARGET_RADIUS, self.COLOR_TARGET)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.TARGET_RADIUS, self.COLOR_TARGET)

        # Draw particles
        for p in self.particles:
            if p['type'] == 'explosion':
                alpha = int(255 * (p['lifespan'] / 30))
                color = self.COLOR_EXPLOSION + (alpha,)
                pos = (int(p['pos'].x), int(p['pos'].y))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)
            elif p['type'] == 'flash':
                alpha = int(255 * (p['lifespan'] / 3))
                color = (255, 255, 255, alpha)
                pos = (int(p['pos'].x), int(p['pos'].y))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)
            elif p['type'] == 'tracer':
                alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
                color = self.COLOR_PROJECTILE + (alpha,)
                start = (int(p['start'].x), int(p['start'].y))
                end = (int(p['end'].x), int(p['end'].y))
                pygame.draw.aaline(self.screen, color, start, end, 2)


        # Draw reticle
        x, y = int(self.reticle_pos.x), int(self.reticle_pos.y)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (x - 10, y), (x + 10, y), 2)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (x, y - 10), (x, y + 10), 2)
        pygame.gfxdraw.aacircle(self.screen, x, y, 8, self.COLOR_RETICLE)

    def _render_ui(self):
        ammo_text = self.font_ui.render(f"AMMO: {self.ammo}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (10, 10))
        
        targets_text = self.font_ui.render(f"TARGETS: {self.targets_destroyed}/{self.NUM_TARGETS}", True, self.COLOR_TEXT)
        text_rect = targets_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(targets_text, text_rect)
        
        if self.game_over:
            if self.targets_destroyed >= self.NUM_TARGETS:
                end_text = self.font_game_over.render("MISSION COMPLETE", True, self.COLOR_RETICLE)
            else:
                end_text = self.font_game_over.render("GAME OVER", True, self.COLOR_TARGET)
            
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ammo": self.ammo,
            "targets_destroyed": self.targets_destroyed,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "human" to see the game being played
    render_mode = "human" # "rgb_array" for no window

    if render_mode == "human":
        GameEnv.metadata["render_modes"].append("human")
        GameEnv.render = lambda self, mode='human': pygame.display.get_surface() is not None and pygame.display.flip()
        GameEnv.reset = (lambda f: lambda self, *args, **kwargs: (f(self, *args, **kwargs), pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT)) if pygame.display.get_surface() is None else None) and f(self, *args, **kwargs))(GameEnv.reset)

    env = GameEnv(render_mode=render_mode)
    obs, info = env.reset()
    
    terminated = False
    total_reward = 0
    
    # --- Manual Control Loop ---
    if render_mode == "human":
        # Create a mapping from Pygame keys to the action space
        key_to_action = {
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
        }
        
        while not terminated:
            movement = 0
            space_held = 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
            
            keys = pygame.key.get_pressed()
            for key, move_action in key_to_action.items():
                if keys[key]:
                    movement = move_action
                    break # Prioritize first key found
            
            if keys[pygame.K_SPACE]:
                space_held = 1

            action = [movement, space_held, 0] # Shift is unused
            
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            terminated = term
            
            env.render()
            env.clock.tick(env.FPS)
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
                # Wait a bit before closing
                pygame.time.wait(2000)

        env.close()