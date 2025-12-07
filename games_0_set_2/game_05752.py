
# Generated: 2025-08-28T05:59:20.668760
# Source Brief: brief_05752.md
# Brief Index: 5752

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = "Controls: Use ← and → to run, and press Space to jump."

    # Must be a short, user-facing description of the game:
    game_description = "Survive a zombie horde for 60 seconds by jumping over them in this fast-paced side-scrolling action game."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (135, 206, 235)  # Light Blue Sky
    COLOR_GROUND = (34, 139, 34)  # Dark Green
    COLOR_PLAYER = (255, 69, 0)    # Bright Red-Orange
    COLOR_ZOMBIE = (60, 179, 113)  # Medium Sea Green
    COLOR_PARTICLE = (255, 255, 255) # White
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_BG = (0, 0, 0, 128) # Semi-transparent black

    # Game physics and properties
    GROUND_Y = 350
    PLAYER_SIZE = (30, 30)
    PLAYER_SPEED = 4
    GRAVITY = 0.6
    JUMP_STRENGTH = -14
    
    ZOMBIE_SIZE = (30, 30)
    INITIAL_ZOMBIE_SPEED = 2.0
    ZOMBIE_SPEED_INCREASE = 0.75 # Amount to increase speed by every 30s
    DIFFICULTY_INTERVAL = 30 * FPS # 30 seconds

    # Zombie spawn rates (in frames)
    INITIAL_SPAWN_INTERVAL = 2 * FPS # 1 every 2 seconds
    FINAL_SPAWN_INTERVAL = 1 * FPS   # 1 every 1 second
    SPAWN_RATE_TRANSITION_FRAMES = 30 * FPS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # State variables are initialized in reset()
        self.player_rect = None
        self.player_vel_y = 0
        self.is_grounded = True
        self.zombies = []
        self.particles = []
        self.zombie_spawn_timer = 0
        self.current_zombie_speed = 0
        self.current_spawn_interval = 0
        self.prev_space_held = False
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_rect = pygame.Rect(self.WIDTH // 4, self.GROUND_Y - self.PLAYER_SIZE[1], *self.PLAYER_SIZE)
        self.player_vel_y = 0
        self.is_grounded = True
        
        self.zombies = []
        self.particles = []
        
        self.zombie_spawn_timer = self.INITIAL_SPAWN_INTERVAL
        self.current_zombie_speed = self.INITIAL_ZOMBIE_SPEED
        self.current_spawn_interval = self.INITIAL_SPAWN_INTERVAL
        
        self.prev_space_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            # If the game is over, do nothing but return the final state
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)
        self._update_player()
        reward = self._update_zombies()
        self._update_particles()

        self.steps += 1
        
        # Continuous survival reward
        reward += 0.01 

        # Check for termination conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if self.game_over:
            reward = -100.0 # Punishment for collision
        elif self.steps >= self.MAX_STEPS:
            reward += 100.0 # Bonus for survival
            self.game_over = True # End game on win

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Horizontal movement
        if movement == 3:  # Left
            self.player_rect.x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_rect.x += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_rect.x = max(0, min(self.player_rect.x, self.WIDTH - self.PLAYER_SIZE[0]))

        # Jumping (on key press)
        if space_held and not self.prev_space_held and self.is_grounded:
            self.player_vel_y = self.JUMP_STRENGTH
            self.is_grounded = False
            # sfx: jump_sound.play()
        
        self.prev_space_held = space_held

    def _update_player(self):
        # Apply gravity
        self.player_vel_y += self.GRAVITY
        self.player_rect.y += self.player_vel_y

        # Ground collision
        if self.player_rect.bottom >= self.GROUND_Y:
            if not self.is_grounded:
                # Create landing particles
                for _ in range(10):
                    self._create_particle(self.player_rect.midbottom)
                # sfx: land_sound.play()

            self.player_rect.bottom = self.GROUND_Y
            self.player_vel_y = 0
            self.is_grounded = True
    
    def _update_zombies(self):
        reward = 0.0

        # --- Difficulty Scaling ---
        # Zombie speed increases in steps
        speed_updates = self.steps // self.DIFFICULTY_INTERVAL
        self.current_zombie_speed = self.INITIAL_ZOMBIE_SPEED + speed_updates * self.ZOMBIE_SPEED_INCREASE

        # Zombie spawn rate increases linearly for the first 30 seconds
        if self.steps < self.SPAWN_RATE_TRANSITION_FRAMES:
            progress = self.steps / self.SPAWN_RATE_TRANSITION_FRAMES
            self.current_spawn_interval = self.INITIAL_SPAWN_INTERVAL - (self.INITIAL_SPAWN_INTERVAL - self.FINAL_SPAWN_INTERVAL) * progress
        else:
            self.current_spawn_interval = self.FINAL_SPAWN_INTERVAL

        # --- Spawning ---
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0:
            zombie_rect = pygame.Rect(self.WIDTH, self.GROUND_Y - self.ZOMBIE_SIZE[1], *self.ZOMBIE_SIZE)
            self.zombies.append({'rect': zombie_rect, 'jumped': False})
            self.zombie_spawn_timer = self.current_spawn_interval + self.np_random.integers(-10, 11) # Add some randomness
            # sfx: zombie_spawn_sound.play()

        # --- Movement and Interaction ---
        for z in self.zombies[:]:
            old_x_center = z['rect'].centerx
            z['rect'].x -= self.current_zombie_speed
            new_x_center = z['rect'].centerx

            # Check for collision
            if self.player_rect.colliderect(z['rect']):
                self.game_over = True
                # sfx: player_hit_sound.play()
                break # Stop processing zombies after a collision

            # Check for successful jump-over reward
            if not z['jumped'] and not self.is_grounded and self.player_rect.bottom < z['rect'].top:
                if old_x_center > self.player_rect.centerx and new_x_center <= self.player_rect.centerx:
                    reward += 1.0
                    z['jumped'] = True
                    # sfx: score_point_sound.play()

            # Remove off-screen zombies
            if z['rect'].right < 0:
                self.zombies.remove(z)
        
        return reward

    def _create_particle(self, pos):
        px, py = pos
        vel_x = self.np_random.uniform(-2, 2)
        vel_y = self.np_random.uniform(-3, -1)
        life = self.np_random.integers(10, 20)
        self.particles.append({'pos': [px, py], 'vel': [vel_x, vel_y], 'life': life, 'max_life': life})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2 # Particle gravity
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
        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

        # Draw particles
        for p in self.particles:
            size = int(3 * (p['life'] / p['max_life']))
            if size > 0:
                pygame.draw.rect(self.screen, self.COLOR_PARTICLE, (int(p['pos'][0]), int(p['pos'][1]), size, size))

        # Draw zombies
        for z in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z['rect'])

        # Draw player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)

    def _render_ui(self):
        # --- Score ---
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topleft=(10, 10))
        # Simple background for readability
        pygame.draw.rect(self.screen, self.COLOR_TEXT_BG, score_rect.inflate(10, 5))
        self.screen.blit(score_text, score_rect)

        # --- Timer ---
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        pygame.draw.rect(self.screen, self.COLOR_TEXT_BG, timer_rect.inflate(10, 5))
        self.screen.blit(timer_text, timer_rect)

        # --- Game Over/Win Message ---
        if self.game_over:
            if self.steps >= self.MAX_STEPS:
                msg = "YOU SURVIVED!"
                color = (0, 255, 127) # Spring Green
            else:
                msg = "GAME OVER"
                color = (255, 0, 0) # Red
            
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            pygame.draw.rect(self.screen, self.COLOR_TEXT_BG, end_rect.inflate(20, 20))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left_seconds": max(0, (self.MAX_STEPS - self.steps) / self.FPS),
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
    import os
    
    env = GameEnv(render_mode="rgb_array")
    
    # For interactive play
    pygame.display.set_caption(env.game_description)
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    terminated = False
    
    # Game loop for human player
    while not terminated:
        movement = 0 # 0=none, 3=left, 4=right
        space_held = 0 # 0=released, 1=held
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        action = [movement, space_held, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

    # Keep the final screen visible for a moment
    pygame.time.wait(2000)
    env.close()
    print(f"Game Over. Final Info: {info}")