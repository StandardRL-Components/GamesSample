
# Generated: 2025-08-27T20:13:21.232900
# Source Brief: brief_02386.md
# Brief Index: 2386

        
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
        "Controls: Use ↑ and ↓ to move your character. Dodge the ghosts and rising tombstones to survive."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape a haunted graveyard in this atmospheric side-scrolling horror game. Dodge procedurally generated obstacles and reach the gate at the end to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    LEVEL_LENGTH = 6000  # Total distance to travel
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (255, 255, 255, 50)
    COLOR_OBSTACLE = (180, 190, 200)
    COLOR_DANGER = (255, 0, 0, 100)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_GATE = (80, 80, 90)

    # Player settings
    PLAYER_WIDTH = 20
    PLAYER_HEIGHT = 40
    PLAYER_SPEED = 8
    PLAYER_X_POS = 100 # Fixed screen position

    # Game dynamics
    INITIAL_SCROLL_SPEED = 3.0
    MAX_SCROLL_SPEED = 8.0
    INITIAL_SPAWN_CHANCE = 0.02
    MAX_SPAWN_CHANCE = 0.05
    
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)

        # Initialize state variables
        self.player_rect = None
        self.scroll_speed = 0
        self.world_x = 0
        self.obstacles = []
        self.bg_elements_far = []
        self.bg_elements_near = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.spawn_chance = 0
        self.checkpoint_reached = False
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Initialize all game state
        self.player_rect = pygame.Rect(self.PLAYER_X_POS, self.SCREEN_HEIGHT / 2 - self.PLAYER_HEIGHT / 2, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        self.scroll_speed = self.INITIAL_SCROLL_SPEED
        self.spawn_chance = self.INITIAL_SPAWN_CHANCE
        self.world_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.checkpoint_reached = False

        self.obstacles.clear()
        self.particles.clear()
        self._generate_background()
        self._generate_particles(100)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, we still need to return a valid tuple
            # The state doesn't change, reward is 0
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self._handle_input(movement)
        self._update_world_and_difficulty()
        self._update_obstacles()
        self._update_particles()
        
        # --- Calculate Reward & Check Termination ---
        reward = 0.1  # Survival reward
        terminated = False

        # Checkpoint reward
        checkpoint_dist = self.LEVEL_LENGTH / 2
        if self.world_x >= checkpoint_dist and not self.checkpoint_reached:
            reward += 10
            self.checkpoint_reached = True
            # sound: checkpoint_sfx()

        # Win/Loss conditions
        if self._check_collision():
            reward = -5.0
            terminated = True
            # sound: player_hit_sfx()
        elif self.world_x >= self.LEVEL_LENGTH and self.player_rect.colliderect(self._get_gate_rect()):
            reward = 100.0
            terminated = True
            # sound: level_win_sfx()
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward
        self.steps += 1
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_particles()
        self._render_gate()
        self._render_obstacles()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance": self.world_x,
        }

    # --- Internal Logic Methods ---
    
    def _handle_input(self, movement):
        if movement == 1:  # Up
            self.player_rect.y -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_rect.y += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_rect.y = max(0, min(self.player_rect.y, self.SCREEN_HEIGHT - self.PLAYER_HEIGHT))

    def _update_world_and_difficulty(self):
        self.world_x += self.scroll_speed

        # Difficulty scaling
        # Increase scroll speed
        if self.steps % 50 == 0 and self.scroll_speed < self.MAX_SCROLL_SPEED:
            self.scroll_speed += 0.05
        # Increase spawn chance
        if self.steps % 100 == 0 and self.spawn_chance < self.MAX_SPAWN_CHANCE:
            self.spawn_chance += 0.001 * (self.FPS / 30) # Scale by framerate, brief says per second

    def _update_obstacles(self):
        # Move existing obstacles
        for obs in self.obstacles:
            obs['rect'].x -= self.scroll_speed
            # Animate ghosts
            if obs['type'] == 'ghost':
                obs['rect'].y = obs['base_y'] + math.sin(self.steps * 0.1 + obs['anim_offset']) * 20

        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['rect'].right > 0]

        # Spawn new obstacles
        if self.np_random.random() < self.spawn_chance and self.world_x < self.LEVEL_LENGTH - self.SCREEN_WIDTH:
            obstacle_type = self.np_random.choice(['ghost', 'tombstone'])
            
            y_pos = self.np_random.integers(0, self.SCREEN_HEIGHT - 80)
            
            # Simple logic to avoid impossible walls
            if self.obstacles:
                last_obs = self.obstacles[-1]
                if abs(last_obs['rect'].x - self.SCREEN_WIDTH) < 200: # If last obs is close
                    if last_obs['rect'].centery < self.SCREEN_HEIGHT / 2:
                        y_pos = self.np_random.integers(self.SCREEN_HEIGHT // 2, self.SCREEN_HEIGHT - 80)
                    else:
                        y_pos = self.np_random.integers(0, self.SCREEN_HEIGHT // 2 - 80)

            if obstacle_type == 'ghost':
                new_obs = {
                    'rect': pygame.Rect(self.SCREEN_WIDTH, y_pos, 40, 40),
                    'type': 'ghost',
                    'base_y': y_pos,
                    'anim_offset': self.np_random.random() * 2 * math.pi
                }
            else: # Tombstone
                height = self.np_random.integers(60, 150)
                y_pos = self.SCREEN_HEIGHT - height
                new_obs = {
                    'rect': pygame.Rect(self.SCREEN_WIDTH, y_pos, 50, height),
                    'type': 'tombstone'
                }
            self.obstacles.append(new_obs)
            # sound: obstacle_spawn_sfx()

    def _check_collision(self):
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs['rect']):
                return True
        return False

    def _get_gate_rect(self):
        gate_x = self.LEVEL_LENGTH - self.world_x
        return pygame.Rect(gate_x, 0, 100, self.SCREEN_HEIGHT)

    # --- Generation Methods ---
    def _generate_background(self):
        self.bg_elements_far.clear()
        self.bg_elements_near.clear()
        for i in range(20): # Far layer (distant trees)
            x = self.np_random.integers(0, self.LEVEL_LENGTH)
            y = self.SCREEN_HEIGHT - self.np_random.integers(80, 150)
            w = self.np_random.integers(10, 20)
            h = self.np_random.integers(80, 150)
            self.bg_elements_far.append(pygame.Rect(x, y, w, h))
        for i in range(40): # Near layer (tombstones)
            x = self.np_random.integers(0, self.LEVEL_LENGTH)
            w = self.np_random.integers(20, 40)
            h = self.np_random.integers(30, 70)
            y = self.SCREEN_HEIGHT - h
            self.bg_elements_near.append(pygame.Rect(x, y, w, h))

    def _generate_particles(self, count):
        self.particles.clear()
        for _ in range(count):
            self.particles.append({
                'x': self.np_random.random() * self.SCREEN_WIDTH,
                'y': self.np_random.random() * self.SCREEN_HEIGHT,
                'speed': self.np_random.random() * 0.5 + 0.2,
                'size': self.np_random.integers(1, 3)
            })

    def _update_particles(self):
        for p in self.particles:
            p['x'] -= p['speed']
            if p['x'] < 0:
                p['x'] = self.SCREEN_WIDTH
                p['y'] = self.np_random.random() * self.SCREEN_HEIGHT

    # --- Rendering Methods ---
    def _render_background(self):
        # Far layer
        for rect in self.bg_elements_far:
            screen_x = rect.x - self.world_x * 0.25
            if -rect.width < screen_x < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, (35, 40, 45), (screen_x, rect.y, rect.width, rect.height))
        # Near layer
        for rect in self.bg_elements_near:
            screen_x = rect.x - self.world_x * 0.5
            if -rect.width < screen_x < self.SCREEN_WIDTH:
                 pygame.draw.rect(self.screen, (50, 55, 60), (screen_x, rect.y, rect.width, rect.height))
                 pygame.draw.rect(self.screen, (60, 65, 70), (screen_x, rect.y, rect.width, 5)) # Top edge highlight

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, (100, 100, 100, 100), (int(p['x']), int(p['y'])), p['size'])

    def _render_player(self):
        # Bobbing animation
        bob = math.sin(self.steps * 0.4) * 3
        player_display_rect = self.player_rect.copy()
        player_display_rect.y += bob

        # Glow effect
        glow_rect = player_display_rect.inflate(20, 20)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect())
        self.screen.blit(glow_surf, glow_rect.topleft)

        # Player body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_display_rect, border_radius=5)
        
    def _render_obstacles(self):
        for obs in self.obstacles:
            # Danger glow
            dist_to_player = math.hypot(obs['rect'].centerx - self.player_rect.centerx, obs['rect'].centery - self.player_rect.centery)
            if dist_to_player < 100:
                glow_intensity = (1 - (dist_to_player / 100))
                glow_radius = int(max(obs['rect'].width, obs['rect'].height) * 0.8)
                s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
                danger_color = (self.COLOR_DANGER[0], self.COLOR_DANGER[1], self.COLOR_DANGER[2], int(self.COLOR_DANGER[3] * glow_intensity))
                pygame.draw.circle(s, danger_color, (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (obs['rect'].centerx - glow_radius, obs['rect'].centery - glow_radius))

            # Obstacle body
            if obs['type'] == 'ghost':
                # Draw a simple ghost shape
                r = obs['rect']
                pygame.gfxdraw.filled_ellipse(self.screen, r.centerx, r.centery, r.width//2, r.height//2, self.COLOR_OBSTACLE)
                pygame.gfxdraw.aacircle(self.screen, r.centerx, r.centery, r.width//2, self.COLOR_OBSTACLE)
            else: # Tombstone
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'], border_top_left_radius=10, border_top_right_radius=10)


    def _render_gate(self):
        gate_rect = self._get_gate_rect()
        if gate_rect.right > 0:
            pygame.draw.rect(self.screen, self.COLOR_GATE, gate_rect)
            # Bars
            for i in range(10):
                bar_x = gate_rect.left + i * 10 + 5
                pygame.draw.line(self.screen, self.COLOR_BG, (bar_x, gate_rect.top), (bar_x, gate_rect.bottom), 3)

    def _render_ui(self):
        distance_text = f"Distance: {int(self.world_x / 100)}m / {int(self.LEVEL_LENGTH / 100)}m"
        score_text = f"Score: {int(self.score)}"
        
        dist_surf = self.font_small.render(distance_text, True, self.COLOR_UI_TEXT)
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(dist_surf, (10, 10))
        self.screen.blit(score_surf, (10, 35))

        if self.game_over:
            won = self.world_x >= self.LEVEL_LENGTH
            message = "YOU ESCAPED!" if won else "YOU DIED"
            color = (150, 255, 150) if won else (255, 100, 100)
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,150))
            self.screen.blit(overlay, (0,0))
            
            msg_surf = self.font_large.render(message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to run the game directly to test it
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Configuration ---
    # Pygame setup for display
    pygame.display.set_caption("Haunted Graveyard Escape")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        movement = 0 # No-op
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Key presses for manual control
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2

        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()
            done = False
        
        if done: # If the episode is over, wait for reset
            action = env.action_space.sample() # The action doesn't matter
            action[0] = 0
        else:
            action = [movement, 0, 0] # Construct the action
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(GameEnv.FPS)

    env.close()