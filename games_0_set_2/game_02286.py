
# Generated: 2025-08-27T19:53:19.113326
# Source Brief: brief_02286.md
# Brief Index: 2286

        
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
    user_guide = "Controls: ←→ to move. Press Space to fire your laser."

    # Must be a short, user-facing description of the game:
    game_description = "A retro arcade shooter. Blast falling blocks with your laser to survive for as long as you can."

    # Frames auto-advance for real-time gameplay
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game Constants
        self.GRID_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.GRID_SIZE
        
        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_BLOCK = (255, 50, 50)
        self.COLOR_LASER = (100, 200, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_WAVE_TEXT = (255, 255, 100)

        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_wave = pygame.font.SysFont("monospace", 40, bold=True)

        # Game parameters
        self.INITIAL_LIVES = 5
        self.MAX_WAVES = 100
        self.MAX_STEPS = 10000
        self.LASER_COOLDOWN_FRAMES = 5
        self.BASE_BLOCK_SPEED = 2.0
        self.BASE_BLOCKS_PER_WAVE = 1
        self.WAVE_TRANSITION_DELAY = 90 # 3 seconds at 30fps

        # Initialize state variables
        self.player_grid_x = 0
        self.player_rect = None
        self.player_lives = 0
        self.current_wave = 0
        self.score = 0
        self.steps = 0
        self.terminated = False
        self.blocks = []
        self.lasers = []
        self.particles = []
        self.laser_cooldown = 0
        self.wave_blocks_to_spawn = 0
        self.wave_spawn_timer = 0
        self.wave_transition_timer = 0
        self.last_dist_to_nearest_block = float('inf')
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_grid_x = self.GRID_WIDTH // 2
        self.player_rect = pygame.Rect(
            self.player_grid_x * self.GRID_SIZE, 
            self.HEIGHT - 2 * self.GRID_SIZE, 
            self.GRID_SIZE, 
            self.GRID_SIZE
        )
        
        self.steps = 0
        self.score = 0
        self.player_lives = self.INITIAL_LIVES
        self.current_wave = 0
        self.terminated = False
        
        self.blocks.clear()
        self.lasers.clear()
        self.particles.clear()
        
        self.laser_cooldown = 0
        self.wave_transition_timer = self.WAVE_TRANSITION_DELAY // 2
        self.wave_blocks_to_spawn = 0

        self.last_dist_to_nearest_block = float('inf')
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        self.terminated = False

        if self.player_lives > 0:
            # --- Pre-action state ---
            dist_before, _ = self._get_nearest_block_dist()

            # --- Handle Input and Update Player ---
            self._handle_input(action)

            # --- Update Game Logic ---
            self._update_lasers()
            self._update_blocks()
            self._update_particles()
            
            # --- Handle Collisions & Event Rewards ---
            reward += self._handle_collisions()
            
            # --- Update Wave Progression ---
            self._update_wave_logic()

            # --- Continuous Rewards ---
            reward += 0.01  # Survival reward per frame
            
            dist_after, _ = self._get_nearest_block_dist()
            if dist_before != float('inf') and dist_after > dist_before:
                reward -= 0.02 # Small penalty for moving away from the action
            self.last_dist_to_nearest_block = dist_after

        # --- Check Termination Conditions ---
        self.steps += 1
        is_victory = self.current_wave > self.MAX_WAVES
        is_defeat = self.player_lives <= 0
        is_timeout = self.steps >= self.MAX_STEPS
        
        if is_victory and not self.terminated:
            reward += 100.0
        if is_defeat and not self.terminated:
            # The -10 penalty is applied in _handle_collisions
            pass

        self.terminated = is_victory or is_defeat or is_timeout
        
        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if movement == 3:  # Left
            self.player_grid_x = max(0, self.player_grid_x - 1)
        elif movement == 4:  # Right
            self.player_grid_x = min(self.GRID_WIDTH - 1, self.player_grid_x + 1)
        
        self.player_rect.x = self.player_grid_x * self.GRID_SIZE

        if self.laser_cooldown > 0:
            self.laser_cooldown -= 1

        if space_held and self.laser_cooldown == 0:
            # SFX: Laser fire
            self.lasers.append(pygame.Rect(self.player_rect.centerx - 2, self.player_rect.top, 4, 10))
            self.laser_cooldown = self.LASER_COOLDOWN_FRAMES

    def _update_lasers(self):
        for laser in self.lasers[:]:
            laser.y -= 15
            if laser.bottom < 0:
                self.lasers.remove(laser)

    def _update_blocks(self):
        speed_multiplier = 1.0 + (self.current_wave // 10) * 0.2
        block_speed = self.BASE_BLOCK_SPEED * speed_multiplier
        
        for block in self.blocks[:]:
            block['pos'][1] += block_speed
            block['rect'].y = int(block['pos'][1])
            if block['rect'].top > self.HEIGHT:
                self.blocks.remove(block)

    def _handle_collisions(self):
        reward = 0
        # Laser-block collisions
        for laser in self.lasers[:]:
            for block in self.blocks[:]:
                if laser.colliderect(block['rect']):
                    # SFX: Explosion
                    self._create_particles(block['rect'].center, self.COLOR_BLOCK)
                    self.lasers.remove(laser)
                    self.blocks.remove(block)
                    self.score += 1
                    reward += 1.0
                    break
        
        # Player-block collisions
        for block in self.blocks[:]:
            if self.player_rect.colliderect(block['rect']):
                # SFX: Player hit
                self.blocks.remove(block)
                self.player_lives -= 1
                reward -= 10.0
                self._create_particles(self.player_rect.center, self.COLOR_PLAYER, 30)
                if self.player_lives <= 0:
                    self.terminated = True
                break
        return reward
        
    def _update_wave_logic(self):
        if self.wave_transition_timer > 0:
            self.wave_transition_timer -= 1
            if self.wave_transition_timer == 0:
                self._start_new_wave()
        
        elif len(self.blocks) == 0 and self.wave_blocks_to_spawn == 0:
            self.wave_transition_timer = self.WAVE_TRANSITION_DELAY

        if self.wave_blocks_to_spawn > 0 and self.wave_spawn_timer <= 0:
            self._spawn_block()
            self.wave_blocks_to_spawn -= 1
            self.wave_spawn_timer = self.np_random.integers(15, 45)

        if self.wave_spawn_timer > 0:
            self.wave_spawn_timer -= 1
            
    def _start_new_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            return
            
        num_blocks = self.BASE_BLOCKS_PER_WAVE + (self.current_wave // 5)
        self.wave_blocks_to_spawn = num_blocks
        self.wave_spawn_timer = 0
    
    def _spawn_block(self):
        grid_x = self.np_random.integers(0, self.GRID_WIDTH)
        rect = pygame.Rect(grid_x * self.GRID_SIZE, -self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        self.blocks.append({'rect': rect, 'pos': [float(rect.x), float(rect.y)]})

    def _get_nearest_block_dist(self):
        if not self.blocks:
            return float('inf'), None
        player_center = self.player_rect.center
        min_dist = float('inf')
        nearest_block = None
        for block in self.blocks:
            dist = math.hypot(block['rect'].centerx - player_center[0], block['rect'].centery - player_center[1])
            if dist < min_dist:
                min_dist = dist
                nearest_block = block
        return min_dist, nearest_block

    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color
            })
            
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            size = max(1, int(p['life'] * 0.2))
            pygame.draw.circle(self.screen, p['color'] + (alpha,), [int(p['pos'][0]), int(p['pos'][1])], size)

        # Render blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, self.COLOR_BLOCK, block['rect'])
            pygame.gfxdraw.rectangle(self.screen, block['rect'], (*self.COLOR_BLOCK, 100))

        # Render lasers
        for laser in self.lasers:
            pygame.draw.rect(self.screen, self.COLOR_LASER, laser)
            glow_rect = laser.inflate(4, 4)
            pygame.gfxdraw.rectangle(self.screen, glow_rect, (*self.COLOR_LASER, 80))

        # Render player
        if self.player_lives > 0:
            # Player as a triangle
            p1 = (self.player_rect.centerx, self.player_rect.top)
            p2 = (self.player_rect.left, self.player_rect.bottom)
            p3 = (self.player_rect.right, self.player_rect.bottom)
            
            # Glow effect
            glow_poly = [(p1[0], p1[1]-2), (p2[0]-2, p2[1]+2), (p3[0]+2, p3[1]+2)]
            pygame.gfxdraw.aapolygon(self.screen, glow_poly, (*self.COLOR_PLAYER, 50))
            pygame.gfxdraw.filled_polygon(self.screen, glow_poly, (*self.COLOR_PLAYER, 50))
            
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3])

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render(f"LIVES: {self.player_lives}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.current_wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH // 2 - wave_text.get_width() // 2, self.HEIGHT - 30))

        # Wave transition text
        if self.wave_transition_timer > self.WAVE_TRANSITION_DELAY / 2:
            wave_label = self.font_wave.render(f"WAVE {self.current_wave}", True, self.COLOR_WAVE_TEXT)
            self.screen.blit(wave_label, (self.WIDTH // 2 - wave_label.get_width() // 2, self.HEIGHT // 2 - wave_label.get_height() // 2))

        # Game Over / Victory Text
        if self.terminated:
            if self.current_wave > self.MAX_WAVES:
                end_text = "VICTORY!"
                color = self.COLOR_WAVE_TEXT
            else:
                end_text = "GAME OVER"
                color = self.COLOR_BLOCK
            
            end_label = self.font_wave.render(end_text, True, color)
            self.screen.blit(end_label, (self.WIDTH // 2 - end_label.get_width() // 2, self.HEIGHT // 2 - end_label.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "lives": self.player_lives,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for display
    pygame.display.set_caption("Block Dodger")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Map keyboard inputs to MultiDiscrete action
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for smooth playback

    print(f"Game Over! Final Info: {info}")
    env.close()