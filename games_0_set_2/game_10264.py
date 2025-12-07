import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:06:54.665604
# Source Brief: brief_00264.md
# Brief Index: 264
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
        "Control geometric clones, morphing between shapes with unique physics to navigate platforms and reach the goal."
    )
    user_guide = (
        "Controls: Use ←→ to move and ↑ to jump. Press ↓ to stomp, space to morph between shapes, and shift to create a new clone."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_EPISODE_STEPS = 5000
    TIME_LIMIT_SECONDS = 60

    # Colors
    COLOR_BG = (25, 28, 40)
    COLOR_PLATFORM = (60, 65, 85)
    COLOR_GOAL = (40, 160, 80)
    COLOR_UI_TEXT = (220, 220, 220)
    
    # Physics
    PLAYER_ACCEL = 0.8
    PLAYER_FRICTION = 0.90
    PLAYER_MAX_SPEED = 6.0
    PLAYER_JUMP_STRENGTH = 13.0
    PLAYER_STOMP_FORCE = 3.0
    MAX_CLONES = 5

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
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20)
            self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_game_over = pygame.font.SysFont(None, 60, bold=True)
            
        # --- Shape Properties ---
        self.SHAPE_PROPS = {
            'cube': {
                'color': (220, 50, 50),
                'gravity': 0.7,
                'bounce': 0.0,
                'size': 24,
            },
            'ball': {
                'color': (50, 100, 220),
                'gravity': 0.5,
                'bounce': 0.6,
                'size': 20,
            }
        }
        self.unlocked_shapes = ['cube', 'ball']
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False
        self.timer = 0
        self.max_y_reached = self.SCREEN_HEIGHT
        
        self.clones = []
        self.active_clone_idx = -1
        self.platforms = []
        self.goal_area = None
        self.particles = []
        
        self.previous_action = np.array([0, 0, 0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False
        self.timer = self.TIME_LIMIT_SECONDS * self.FPS
        self.max_y_reached = self.SCREEN_HEIGHT
        
        self.clones = []
        self.particles = []
        self.previous_action = np.array([0, 0, 0])
        
        self._generate_level()
        self._create_clone(pygame.Vector2(100, 200), self.unlocked_shapes[0])
        self.active_clone_idx = 0
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms = [
            pygame.Rect(0, self.SCREEN_HEIGHT - 20, self.SCREEN_WIDTH, 20),
            pygame.Rect(200, 300, 150, 15),
            pygame.Rect(400, 220, 100, 15),
            pygame.Rect(150, 150, 80, 15),
        ]
        self.goal_area = pygame.Rect(self.SCREEN_WIDTH - 80, 50, 60, 30)

    def _create_clone(self, pos, shape_name):
        if len(self.clones) >= self.MAX_CLONES:
            return False
            
        shape_props = self.SHAPE_PROPS[shape_name]
        clone = {
            'pos': pygame.Vector2(pos),
            'vel': pygame.Vector2(0, 0),
            'shape': shape_name,
            'size': shape_props['size'],
            'on_ground': False,
            'rect': pygame.Rect(pos.x, pos.y, shape_props['size'], shape_props['size'])
        }
        self.clones.append(clone)
        # Sound: Clone created
        self._create_particles(pos, 20, shape_props['color'], 2.5)
        return True

    def step(self, action):
        reward = 0.0
        self.steps += 1
        self.timer -= 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        reward += self._handle_input(action)
        self.previous_action = action
        
        # --- Update Physics and State ---
        self._update_clones()
        self._update_particles()
        
        # --- Calculate Rewards & Check Termination ---
        term_reward, terminated = self._check_termination_and_get_rewards()
        reward += term_reward
        
        # Survival/time rewards
        reward -= 0.01  # Time penalty
        if not terminated:
            reward += 0.01 * len(self.clones) # Small reward for each surviving clone
        
        # New height reward
        current_max_y = min(c['pos'].y for c in self.clones) if self.clones else self.SCREEN_HEIGHT
        if current_max_y < self.max_y_reached:
            reward += 5.0
            self.max_y_reached = current_max_y
            
        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        reward = 0
        if not self.clones or self.active_clone_idx >= len(self.clones):
            return 0
            
        active_clone = self.clones[self.active_clone_idx]
        movement, space_action, shift_action = action[0], action[1], action[2]
        
        # Movement
        if movement == 1 and active_clone['on_ground']: # Up (Jump)
            active_clone['vel'].y = -self.PLAYER_JUMP_STRENGTH
            active_clone['on_ground'] = False
            # Sound: Jump
            self._create_particles(active_clone['pos'] + (active_clone['size']/2, active_clone['size']), 5, (200,200,200), 1.5)
        elif movement == 2: # Down (Stomp)
            active_clone['vel'].y += self.PLAYER_STOMP_FORCE
        elif movement == 3: # Left
            active_clone['vel'].x -= self.PLAYER_ACCEL
        elif movement == 4: # Right
            active_clone['vel'].x += self.PLAYER_ACCEL
            
        # Morph (Spacebar press)
        if space_action == 1 and self.previous_action[1] == 0:
            current_shape_idx = self.unlocked_shapes.index(active_clone['shape'])
            next_shape_idx = (current_shape_idx + 1) % len(self.unlocked_shapes)
            new_shape = self.unlocked_shapes[next_shape_idx]
            
            active_clone['shape'] = new_shape
            active_clone['size'] = self.SHAPE_PROPS[new_shape]['size']
            # Sound: Morph
            self._create_particles(active_clone['pos'] + (active_clone['size']/2, active_clone['size']/2), 30, self.SHAPE_PROPS[new_shape]['color'], 3.0)

        # Clone (Shift press)
        if shift_action == 1 and self.previous_action[2] == 0:
            if self._create_clone(active_clone['pos'], active_clone['shape']):
                self.active_clone_idx = len(self.clones) - 1
                reward += 1.0

        return reward

    def _update_clones(self):
        clones_to_remove = []
        for i, clone in enumerate(self.clones):
            shape_props = self.SHAPE_PROPS[clone['shape']]
            
            # Apply forces
            clone['vel'].y += shape_props['gravity']
            clone['vel'].x *= self.PLAYER_FRICTION
            clone['vel'].x = max(-self.PLAYER_MAX_SPEED, min(self.PLAYER_MAX_SPEED, clone['vel'].x))
            
            # --- Collision Detection (separated axes) ---
            clone['on_ground'] = False
            
            # Horizontal
            clone['pos'].x += clone['vel'].x
            clone['rect'].x = int(clone['pos'].x)
            for plat in self.platforms:
                if clone['rect'].colliderect(plat):
                    if clone['vel'].x > 0: # Moving right
                        clone['rect'].right = plat.left
                    elif clone['vel'].x < 0: # Moving left
                        clone['rect'].left = plat.right
                    clone['pos'].x = clone['rect'].x
                    clone['vel'].x = 0
            
            # Vertical
            clone['pos'].y += clone['vel'].y
            clone['rect'].y = int(clone['pos'].y)
            for plat in self.platforms:
                if clone['rect'].colliderect(plat):
                    if clone['vel'].y > 0: # Moving down
                        clone['rect'].bottom = plat.top
                        clone['on_ground'] = True
                        if clone['vel'].y > 2.0: # Landing particle effect
                            self._create_particles(pygame.Vector2(clone['rect'].centerx, clone['rect'].bottom), 3, (180,180,180), 1.0)
                        clone['vel'].y *= -shape_props['bounce']
                        if abs(clone['vel'].y) < 1: clone['vel'].y = 0
                    elif clone['vel'].y < 0: # Moving up
                        clone['rect'].top = plat.bottom
                        clone['vel'].y *= -shape_props['bounce']
                    clone['pos'].y = clone['rect'].y

            # Screen bounds
            if clone['pos'].x < 0:
                clone['pos'].x = 0
                clone['vel'].x = 0
            if clone['pos'].x > self.SCREEN_WIDTH - clone['size']:
                clone['pos'].x = self.SCREEN_WIDTH - clone['size']
                clone['vel'].x = 0
            
            if clone['pos'].y > self.SCREEN_HEIGHT:
                clones_to_remove.append(i)
        
        # Remove clones that fell off
        for i in sorted(clones_to_remove, reverse=True):
            if i == self.active_clone_idx:
                # If active clone is removed, find a new one or set to -1
                self.active_clone_idx = (len(self.clones) - 2) % len(self.clones) if len(self.clones) > 1 else -1
            elif i < self.active_clone_idx:
                self.active_clone_idx -= 1
            del self.clones[i]

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _check_termination_and_get_rewards(self):
        # Win condition
        for clone in self.clones:
            if self.goal_area.colliderect(clone['rect']):
                self.game_won = True
                # Sound: Win
                return 100.0, True
        
        # Lose conditions
        if not self.clones:
            # Sound: Fail
            return -100.0, True
        if self.timer <= 0:
            return -100.0, True
        if self.steps >= self.MAX_EPISODE_STEPS:
            return 0.0, True
            
        return 0.0, False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Goal
        goal_surface = pygame.Surface(self.goal_area.size, pygame.SRCALPHA)
        goal_surface.fill((*self.COLOR_GOAL, 100))
        self.screen.blit(goal_surface, self.goal_area.topleft)
        pygame.draw.rect(self.screen, self.COLOR_GOAL, self.goal_area, 2)

        # Render Platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)
            
        # Render Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            radius = int(p['radius'] * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'].x), int(p['pos'].y), radius, (*p['color'], alpha)
                )

        # Render Clones
        for i, clone in enumerate(self.clones):
            props = self.SHAPE_PROPS[clone['shape']]
            clone_rect = pygame.Rect(int(clone['pos'].x), int(clone['pos'].y), clone['size'], clone['size'])
            
            # Active clone indicator (glow)
            if i == self.active_clone_idx and not self.game_over:
                glow_size = clone['size'] + 10 + 4 * math.sin(self.steps * 0.2)
                glow_alpha = 70 + 30 * math.sin(self.steps * 0.2)
                glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
                
                if clone['shape'] == 'cube':
                     pygame.draw.rect(glow_surf, (255, 255, 255, glow_alpha), glow_surf.get_rect(), border_radius=4)
                else: # ball
                     pygame.draw.circle(glow_surf, (255, 255, 255, glow_alpha), (glow_size/2, glow_size/2), glow_size/2)
                self.screen.blit(glow_surf, (clone_rect.centerx - glow_size/2, clone_rect.centery - glow_size/2))

            # Draw shape
            if clone['shape'] == 'cube':
                pygame.draw.rect(self.screen, props['color'], clone_rect, border_radius=2)
            else: # ball
                pygame.gfxdraw.filled_circle(self.screen, clone_rect.centerx, clone_rect.centery, int(clone['size']/2), props['color'])
                pygame.gfxdraw.aacircle(self.screen, clone_rect.centerx, clone_rect.centery, int(clone['size']/2), props['color'])

    def _render_ui(self):
        # Timer
        time_str = f"Time: {max(0, self.timer // self.FPS):02d}"
        time_surf = self.font_ui.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - time_surf.get_width() - 10, 10))
        
        # Clones
        clone_str = f"Clones: {len(self.clones)}/{self.MAX_CLONES}"
        clone_surf = self.font_ui.render(clone_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(clone_surf, (10, 10))

        # Game Over Text
        if self.game_over:
            msg = "GOAL!" if self.game_won else "FAILED"
            color = self.COLOR_GOAL if self.game_won else (220, 50, 50)
            end_surf = self.font_game_over.render(msg, True, color)
            self.screen.blit(end_surf, end_surf.get_rect(center=self.screen.get_rect().center))
            
    def _create_particles(self, pos, count, color, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, max_speed)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            life = random.randint(15, 30)
            self.particles.append({
                'pos': pygame.Vector2(pos), 'vel': vel, 'life': life,
                'max_life': life, 'color': color, 'radius': random.randint(2, 5)
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "clones": len(self.clones),
            "game_won": self.game_won
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # The original __main__ block is for interactive testing and requires a display.
    # It's not compatible with the headless environment required for automated testing.
    # We can keep it for developers who might want to run the game visually.
    try:
        os.environ.pop("SDL_VIDEODRIVER") # Allow display for manual play
        env = GameEnv()
        env.reset()
        
        pygame.display.set_caption("Geometric Clone Acrobatics")
        render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        
        action = np.array([0, 0, 0]) # none, no-space, no-shift
        done = False
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            
            # Action mapping
            movement = 0 # None
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = np.array([movement, space, shift])
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            render_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(env.FPS)
            
        env.close()
    except pygame.error as e:
        print(f"Could not run interactive test, likely due to headless environment: {e}")
        # Fallback to a simple non-visual test
        env = GameEnv()
        env.validate_implementation()
        env.close()