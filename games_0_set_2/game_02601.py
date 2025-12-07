
# Generated: 2025-08-28T05:22:24.700785
# Source Brief: brief_02601.md
# Brief Index: 2601

        
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
        "Controls: Arrow keys to move. Avoid monsters and collect hearts to survive."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a 60-second onslaught of grid-based monsters while collecting hearts to maintain your health in this procedurally generated horror experience."
    )

    # Frames auto-advance for time-based gameplay and smooth animations.
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 32
    GRID_HEIGHT = 20
    CELL_SIZE = 20
    
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    
    LOGICAL_TICK_RATE = 10 # Player and monster logic updates 10 times per second
    LOGICAL_TICK_FRAMES = FPS // LOGICAL_TICK_RATE
    
    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_GRID = (30, 30, 50)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_HEART = (255, 105, 180)
    COLOR_HEART_GLOW = (255, 105, 180, 60)
    MONSTER_COLORS = [(255, 0, 0), (255, 0, 255), (255, 165, 0)]
    MONSTER_GLOW_COLORS = [(255, 0, 0, 50), (255, 0, 255, 50), (255, 165, 0, 50)]
    COLOR_TEXT = (240, 240, 240)
    COLOR_DAMAGE_FLASH = (255, 0, 0, 100)
    
    MAX_HEALTH = 5
    START_HEALTH = 3
    
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
        self.font_ui = pygame.font.Font(None, 32)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_grid_pos = [0, 0]
        self.player_visual_pos = [0.0, 0.0]
        self.player_health = 0
        self.monsters = []
        self.hearts = []
        self.particles = []
        self.screen_flash_timer = 0
        self.last_monster_spawn_time = 0
        self.last_speed_increase_time = 0
        self.monster_speed_multiplier = 1.0
        self.max_monsters = 1
        self.last_action_time = 0
        self.np_random = None

        self.validate_implementation()

    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to center pixel coordinates."""
        px = (grid_pos[0] + 0.5) * self.CELL_SIZE
        py = (grid_pos[1] + 0.5) * self.CELL_SIZE
        return [px, py]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Player
        self.player_grid_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.player_visual_pos = self._grid_to_pixel(self.player_grid_pos)
        self.player_health = self.START_HEALTH
        self.last_action_time = 0

        # Monsters
        self.monsters = []
        self.last_monster_spawn_time = 0
        self.last_speed_increase_time = 0
        self.monster_speed_multiplier = 1.0
        self.max_monsters = 1
        self._spawn_monster()
        
        # Hearts
        self.hearts = []
        potential_heart_locs = [
            (2, 2), (self.GRID_WIDTH - 3, 2),
            (2, self.GRID_HEIGHT - 3), (self.GRID_WIDTH - 3, self.GRID_HEIGHT - 3),
            (self.GRID_WIDTH // 2, 2)
        ]
        for loc in potential_heart_locs:
            self.hearts.append({
                "grid_pos": list(loc),
                "visual_pos": self._grid_to_pixel(loc),
                "active": True,
                "respawn_timer": 0
            })

        # Effects
        self.particles = []
        self.screen_flash_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.01  # Small reward for surviving each frame
        
        if not self.game_over:
            # --- LOGIC UPDATES ---
            
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            
            # Process input on a logical tick
            if self.steps >= self.last_action_time + self.LOGICAL_TICK_FRAMES:
                self.last_action_time = self.steps
                self._handle_input(movement)

            # Update game elements
            self._update_player_visuals()
            self._update_monsters()
            self._update_hearts()
            
            # Check for collisions and apply rewards/penalties
            collision_reward = self._check_collisions()
            reward += collision_reward
            
            # Update difficulty over time
            self._update_difficulty()
            
            # Update effects
            self._update_particles()
            if self.screen_flash_timer > 0:
                self.screen_flash_timer -= 1
        
        self.steps += 1
        self.score += reward
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.player_health <= 0:
                reward -= 100 # Large penalty for dying
            else: # Survived 60 seconds
                reward += 50
            self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement):
        """Updates player's target grid position based on action."""
        target_pos = list(self.player_grid_pos)
        if movement == 1: # Up
            target_pos[1] -= 1
        elif movement == 2: # Down
            target_pos[1] += 1
        elif movement == 3: # Left
            target_pos[0] -= 1
        elif movement == 4: # Right
            target_pos[0] += 1
        
        # Clamp to grid boundaries
        target_pos[0] = max(0, min(self.GRID_WIDTH - 1, target_pos[0]))
        target_pos[1] = max(0, min(self.GRID_HEIGHT - 1, target_pos[1]))
        
        self.player_grid_pos = target_pos

    def _update_player_visuals(self):
        """Interpolates player's visual position towards its grid position."""
        target_pixel_pos = self._grid_to_pixel(self.player_grid_pos)
        for i in range(2):
            self.player_visual_pos[i] += (target_pixel_pos[i] - self.player_visual_pos[i]) * 0.5

    def _update_monsters(self):
        """Updates monster movement and logic."""
        for monster in self.monsters:
            if self.steps >= monster['next_move_time']:
                monster['next_move_time'] = self.steps + int(self.LOGICAL_TICK_FRAMES / self.monster_speed_multiplier)
                
                # Simple movement patterns
                if monster['type'] == 0: # Pacer
                    next_pos = [monster['grid_pos'][0] + monster['dir'][0], monster['grid_pos'][1] + monster['dir'][1]]
                    if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT) or self.np_random.random() < 0.1:
                        monster['dir'] = self.np_random.choice([(-1,0), (1,0), (0,-1), (0,1)])
                    else:
                        monster['grid_pos'] = next_pos
                
                elif monster['type'] == 1: # Circler
                    path = [(0,0), (1,0), (1,1), (0,1)]
                    monster['path_idx'] = (monster['path_idx'] + 1) % len(path)
                    offset = path[monster['path_idx']]
                    monster['grid_pos'] = [monster['origin'][0] + offset[0], monster['origin'][1] + offset[1]]
            
            # Interpolate visual position
            target_pixel_pos = self._grid_to_pixel(monster['grid_pos'])
            for i in range(2):
                monster['visual_pos'][i] += (target_pixel_pos[i] - monster['visual_pos'][i]) * 0.3

    def _update_hearts(self):
        """Updates heart respawn timers."""
        for heart in self.hearts:
            if not heart['active']:
                heart['respawn_timer'] -= 1
                if heart['respawn_timer'] <= 0:
                    heart['active'] = True
                    # Heart collected sfx placeholder
    
    def _update_difficulty(self):
        """Increases monster count and speed over time."""
        current_time_sec = self.steps / self.FPS
        
        # Increase monster count every 10 seconds
        if current_time_sec > self.last_monster_spawn_time + 10 and self.max_monsters < 10:
            self.last_monster_spawn_time = current_time_sec
            self.max_monsters += 1
        
        # Spawn new monsters up to the max
        if len(self.monsters) < self.max_monsters:
            self._spawn_monster()
            
        # Increase monster speed every 10 seconds
        if current_time_sec > self.last_speed_increase_time + 10:
            self.last_speed_increase_time = current_time_sec
            self.monster_speed_multiplier += 0.05
    
    def _spawn_monster(self):
        """Spawns a new monster at a random location."""
        while True:
            pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]
            # Don't spawn on player
            if math.dist(pos, self.player_grid_pos) > 5:
                break
        
        m_type = self.np_random.integers(0, 2)
        color_idx = self.np_random.integers(0, len(self.MONSTER_COLORS))
        
        monster = {
            "grid_pos": pos,
            "visual_pos": self._grid_to_pixel(pos),
            "type": m_type,
            "color": self.MONSTER_COLORS[color_idx],
            "glow_color": self.MONSTER_GLOW_COLORS[color_idx],
            "next_move_time": self.steps,
        }
        if m_type == 0: # Pacer
            monster['dir'] = self.np_random.choice([(-1,0), (1,0), (0,-1), (0,1)])
        elif m_type == 1: # Circler
            monster['origin'] = pos
            monster['path_idx'] = 0
            
        self.monsters.append(monster)
        
    def _check_collisions(self):
        """Checks for player collisions with monsters and hearts."""
        reward = 0
        
        # Monster collisions
        for monster in self.monsters:
            if self.player_grid_pos == monster['grid_pos']:
                self.player_health -= 1
                reward -= 5
                self.screen_flash_timer = 5 # Flash for 5 frames
                # Player damage sfx placeholder
                # Remove monster on collision to give player a break
                self.monsters.remove(monster)
                break # Only one collision per frame
                
        # Heart collisions
        for heart in self.hearts:
            if heart['active'] and self.player_grid_pos == heart['grid_pos']:
                self.player_health = min(self.MAX_HEALTH, self.player_health + 1)
                reward += 5
                heart['active'] = False
                heart['respawn_timer'] = 15 * self.LOGICAL_TICK_RATE # 15 logical steps
                self._create_particles(heart['visual_pos'], self.COLOR_HEART, 20)
                # Heart collect sfx placeholder
        
        return reward
        
    def _check_termination(self):
        """Checks for game over conditions."""
        return self.player_health <= 0 or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the main game elements (grid, entities)."""
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw hearts
        for heart in self.hearts:
            if heart['active']:
                pos = (int(heart['visual_pos'][0]), int(heart['visual_pos'][1]))
                radius = int(self.CELL_SIZE * 0.4)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 4, self.COLOR_HEART_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_HEART)

        # Draw monsters
        for monster in self.monsters:
            pos = (int(monster['visual_pos'][0]), int(monster['visual_pos'][1]))
            size = int(self.CELL_SIZE * 0.8)
            rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
            
            # Glow
            glow_rect = rect.inflate(8, 8)
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, monster['glow_color'], shape_surf.get_rect(), border_radius=3)
            self.screen.blit(shape_surf, glow_rect.topleft)
            
            # Body
            pygame.draw.rect(self.screen, monster['color'], rect, border_radius=3)

        # Draw player
        player_pos = (int(self.player_visual_pos[0]), int(self.player_visual_pos[1]))
        size = int(self.CELL_SIZE * 0.9)
        rect = pygame.Rect(player_pos[0] - size // 2, player_pos[1] - size // 2, size, size)
        
        # Glow
        glow_rect = rect.inflate(10, 10)
        shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, self.COLOR_PLAYER_GLOW, shape_surf.get_rect(), border_radius=5)
        self.screen.blit(shape_surf, glow_rect.topleft)
        
        # Body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=5)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['radius']))

    def _render_ui(self):
        """Renders UI elements like health, timer, and game over text."""
        # Health bar
        for i in range(self.MAX_HEALTH):
            heart_icon_pos = (20 + i * 25, 25)
            if i < self.player_health:
                color = self.COLOR_HEART
            else:
                color = (50, 50, 50)
            pygame.gfxdraw.filled_circle(self.screen, heart_icon_pos[0], heart_icon_pos[1], 8, color)
        
        # Timer
        time_left = max(0, self.GAME_DURATION_SECONDS - (self.steps / self.FPS))
        timer_text = f"{time_left:.1f}"
        text_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(text_surf, text_rect)
        
        # Damage flash
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill(self.COLOR_DAMAGE_FLASH)
            self.screen.blit(flash_surface, (0, 0))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.player_health <= 0:
                msg = "YOU DIED"
            else:
                msg = "YOU SURVIVED!"
            
            text_surf = self.font_game_over.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _create_particles(self, pos, color, count):
        """Creates a burst of particles."""
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': self.np_random.random() * 2 + 1,
                'life': self.np_random.integers(10, 20),
                'color': color,
            })

    def _update_particles(self):
        """Updates position and lifetime of particles."""
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "time_left": max(0, self.GAME_DURATION_SECONDS - (self.steps / self.FPS)),
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
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

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # Set up a window to view the game
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get keyboard input for human play
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        # The action space is MultiDiscrete, but we only use the first part for this game
        action = [movement, 0, 0]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(env.FPS)

    env.close()