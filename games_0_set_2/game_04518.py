
# Generated: 2025-08-28T02:39:26.463740
# Source Brief: brief_04518.md
# Brief Index: 4518

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the crosshair. Press space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down target practice game. Hit all targets in each stage with limited ammo and time. "
        "Difficulty increases with each stage."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30  # Frames per second for game logic
    FONT_SIZE_LARGE = 48
    FONT_SIZE_MEDIUM = 24
    FONT_SIZE_SMALL = 18

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_PLAYER = (0, 255, 255) # Cyan
    COLOR_PROJECTILE = (100, 255, 100) # Bright Green
    COLOR_TARGET = (255, 50, 50) # Bright Red
    COLOR_HIT_MARKER = (255, 255, 0) # Yellow
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0)

    # Game Parameters
    INITIAL_AMMO = 10
    INITIAL_TIME_SECONDS = 60
    NUM_TARGETS = 15
    TOTAL_STAGES = 3
    CROSSHAIR_SPEED = 8
    PROJECTILE_SPEED = 15
    FIRE_COOLDOWN_FRAMES = 6 # 5 shots/sec max
    PARTICLE_LIFESPAN = 20
    HIT_MARKER_LIFESPAN = 45

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, self.FONT_SIZE_LARGE)
        self.font_medium = pygame.font.Font(None, self.FONT_SIZE_MEDIUM)
        self.font_small = pygame.font.Font(None, self.FONT_SIZE_SMALL)
        
        # Etc...        
        self.np_random = None
        
        # Initialize state variables
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_condition = False
        self.current_stage = 1
        self.stage_respawn_time = None
        self.ammo = 0
        self.time_remaining = 0
        self.crosshair_pos = [0, 0]
        self.targets = []
        self.projectiles = []
        self.particles = []
        self.hit_markers = []
        self.last_shot_frame = 0
        self.targets_hit_this_stage = 0
        self.step_reward = 0.0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.current_stage = 1
        
        self.particles = []
        self.projectiles = []
        
        self._start_new_stage()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _start_new_stage(self):
        self.ammo = self.INITIAL_AMMO
        self.time_remaining = self.INITIAL_TIME_SECONDS * self.FPS
        self.crosshair_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        self.last_shot_frame = -self.FIRE_COOLDOWN_FRAMES
        self.targets_hit_this_stage = 0
        self.hit_markers = []
        
        if self.current_stage == 1:
            self.stage_respawn_time = None
        elif self.current_stage == 2:
            self.stage_respawn_time = 5 * self.FPS
        else: # Stage 3
            self.stage_respawn_time = 3 * self.FPS

        self.targets = []
        for _ in range(self.NUM_TARGETS):
            self._spawn_target()

    def _spawn_target(self, position=None):
        if position is None:
            # Ensure targets are not too close to edges or each other
            while True:
                x = self.np_random.integers(40, self.SCREEN_WIDTH - 40)
                y = self.np_random.integers(40, self.SCREEN_HEIGHT - 80)
                pos = [x, y]
                
                # Check distance to other targets
                too_close = False
                for t in self.targets:
                    if math.hypot(pos[0] - t['pos'][0], pos[1] - t['pos'][1]) < 30:
                        too_close = True
                        break
                if not too_close:
                    break
        else:
            pos = position
        
        self.targets.append({
            'pos': pos,
            'radius': 10,
            'active': True,
            'destroyed_time': -1
        })
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Update game logic
        self.steps += 1
        self.step_reward = 0.0

        self._handle_input(action)
        self._update_game_state()
        
        terminated = self._check_termination()
        reward = self.step_reward
        
        if terminated:
            self.game_over = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Move crosshair
        if movement == 1: self.crosshair_pos[1] -= self.CROSSHAIR_SPEED
        if movement == 2: self.crosshair_pos[1] += self.CROSSHAIR_SPEED
        if movement == 3: self.crosshair_pos[0] -= self.CROSSHAIR_SPEED
        if movement == 4: self.crosshair_pos[0] += self.CROSSHAIR_SPEED
        
        self.crosshair_pos[0] = np.clip(self.crosshair_pos[0], 0, self.SCREEN_WIDTH)
        self.crosshair_pos[1] = np.clip(self.crosshair_pos[1], 0, self.SCREEN_HEIGHT)

        # Fire projectile
        if space_held and self.ammo > 0 and (self.steps - self.last_shot_frame) > self.FIRE_COOLDOWN_FRAMES:
            # SFX: Laser shot
            self.last_shot_frame = self.steps
            self.ammo -= 1
            
            # Calculate projectile velocity vector
            target_pos = self.crosshair_pos
            start_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20]
            angle = math.atan2(target_pos[1] - start_pos[1], target_pos[0] - start_pos[0])
            
            self.projectiles.append({
                'pos': list(start_pos),
                'vel': [math.cos(angle) * self.PROJECTILE_SPEED, math.sin(angle) * self.PROJECTILE_SPEED]
            })

    def _update_game_state(self):
        # Update timer
        self.time_remaining = max(0, self.time_remaining - 1)

        # Update projectiles
        for proj in self.projectiles[:]:
            proj['pos'][0] += proj['vel'][0]
            proj['pos'][1] += proj['vel'][1]
            
            hit_target = False
            for target in self.targets:
                if target['active']:
                    dist = math.hypot(proj['pos'][0] - target['pos'][0], proj['pos'][1] - target['pos'][1])
                    if dist < target['radius'] + 4: # 4 is projectile radius
                        # SFX: Target explosion
                        target['active'] = False
                        target['destroyed_time'] = self.steps
                        self.score += 1
                        self.step_reward += 1.0
                        self.targets_hit_this_stage += 1
                        self.hit_markers.append({'pos': list(target['pos']), 'life': self.HIT_MARKER_LIFESPAN})
                        self._create_explosion(target['pos'], self.COLOR_TARGET)
                        self.projectiles.remove(proj)
                        hit_target = True
                        break # Projectile can only hit one target
            
            if hit_target:
                continue

            # Check for wall collision
            if not (0 < proj['pos'][0] < self.SCREEN_WIDTH and 0 < proj['pos'][1] < self.SCREEN_HEIGHT):
                # SFX: Ricochet or fizzle
                self.projectiles.remove(proj)
                self.step_reward -= 0.1 # Penalty for missing

        # Update target respawns
        if self.stage_respawn_time is not None:
            for target in self.targets:
                if not target['active'] and (self.steps - target['destroyed_time']) > self.stage_respawn_time:
                    target['active'] = True
                    self.targets_hit_this_stage -= 1 # Re-add to the pool of targets to hit

        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # Update hit markers
        for hm in self.hit_markers[:]:
            hm['life'] -= 1
            if hm['life'] <= 0:
                self.hit_markers.remove(hm)
    
    def _check_termination(self):
        # Stage clear condition
        if self.targets_hit_this_stage >= self.NUM_TARGETS:
            self.step_reward += 10.0 # Stage clear bonus
            if self.current_stage == self.TOTAL_STAGES:
                self.step_reward += 50.0 # Game win bonus
                self.win_condition = True
                return True
            else:
                self.current_stage += 1
                self._start_new_stage()
                return False # Not terminal, just advancing stage

        # Loss conditions
        # Out of time
        if self.time_remaining <= 0:
            self.step_reward -= 50.0
            return True
            
        # Out of ammo and no more active projectiles
        if self.ammo <= 0 and not self.projectiles:
            # Check if all targets are hit, if so, it's a win, not a loss
            if self.targets_hit_this_stage < self.NUM_TARGETS:
                self.step_reward -= 50.0
                return True

        return False
    
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
        # Draw grid
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Draw hit markers
        for hm in self.hit_markers:
            alpha = int(255 * (hm['life'] / self.HIT_MARKER_LIFESPAN))
            if alpha > 0:
                pos = (int(hm['pos'][0]), int(hm['pos'][1]))
                size = 12
                line_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.line(line_surf, (*self.COLOR_HIT_MARKER, alpha), (0, 0), (size*2, size*2), 3)
                pygame.draw.line(line_surf, (*self.COLOR_HIT_MARKER, alpha), (0, size*2), (size*2, 0), 3)
                self.screen.blit(line_surf, (pos[0] - size, pos[1] - size))

        # Draw targets
        for target in self.targets:
            if target['active']:
                pos = (int(target['pos'][0]), int(target['pos'][1]))
                radius = int(target['radius'])
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_TARGET)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_TARGET)

        # Draw particles
        for p in self.particles:
            alpha = max(0, int(255 * (p['life'] / self.PARTICLE_LIFESPAN)))
            if alpha > 0:
                color = (*p['color'], alpha)
                size = max(1, int(p['life'] / self.PARTICLE_LIFESPAN * 4))
                # Use a surface for alpha blending
                particle_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color, (size, size), size)
                self.screen.blit(particle_surf, (int(p['pos'][0]) - size, int(p['pos'][1]) - size))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, self.COLOR_PROJECTILE)
            
        # Draw player base
        base_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 10)
        pygame.draw.rect(self.screen, (100, 100, 120), (base_pos[0]-20, base_pos[1]-10, 40, 20), border_radius=5)
        pygame.draw.circle(self.screen, (150, 150, 170), base_pos, 10)

        # Draw crosshair
        if not self.game_over:
            cx, cy = int(self.crosshair_pos[0]), int(self.crosshair_pos[1])
            size = 12
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (cx - size, cy), (cx + size, cy), 2)
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (cx, cy - size), (cx, cy + size), 2)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, size, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, cx, cy, size//2, self.COLOR_PLAYER)

    def _render_text(self, text, font, pos, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, align="center"):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        
        if align == "center":
            text_rect = text_surf.get_rect(center=pos)
            shadow_rect = shadow_surf.get_rect(center=(pos[0]+2, pos[1]+2))
        elif align == "left":
            text_rect = text_surf.get_rect(midleft=pos)
            shadow_rect = shadow_surf.get_rect(midleft=(pos[0]+2, pos[1]+2))
        elif align == "right":
            text_rect = text_surf.get_rect(midright=pos)
            shadow_rect = shadow_surf.get_rect(midright=(pos[0]+2, pos[1]+2))

        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        self._render_text(f"SCORE: {self.score}", self.font_medium, (20, 30), align="left")
        # Ammo
        self._render_text(f"AMMO: {self.ammo}", self.font_medium, (self.SCREEN_WIDTH / 2, 30), align="center")
        # Time
        time_sec = math.ceil(self.time_remaining / self.FPS)
        time_color = self.COLOR_TEXT if time_sec > 10 else self.COLOR_TARGET
        self._render_text(f"TIME: {time_sec}", self.font_medium, (self.SCREEN_WIDTH - 20, 30), color=time_color, align="right")
        # Stage
        self._render_text(f"STAGE: {self.current_stage}/{self.TOTAL_STAGES}", self.font_small, (self.SCREEN_WIDTH / 2, 60), align="center")

        # Game Over / Win Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.win_condition:
                self._render_text("MISSION COMPLETE", self.font_large, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 30))
                self._render_text(f"FINAL SCORE: {self.score}", self.font_medium, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 30))
            else:
                self._render_text("GAME OVER", self.font_large, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 30))
                self._render_text(f"FINAL SCORE: {self.score}", self.font_medium, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 30))

    def _create_explosion(self, pos, color):
        num_particles = 30
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            p_color = (
                max(0, min(255, color[0] + self.np_random.integers(-20, 20))),
                max(0, min(255, color[1] + self.np_random.integers(-20, 20))),
                max(0, min(255, color[2] + self.np_random.integers(-20, 20)))
            )
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(self.PARTICLE_LIFESPAN // 2, self.PARTICLE_LIFESPAN),
                'color': p_color
            })
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "ammo": self.ammo,
            "time_remaining_seconds": math.ceil(self.time_remaining / self.FPS)
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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Target Practice")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Event Handling ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Game Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already the rendered screen, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()