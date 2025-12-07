import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your character. "
        "Collect the cyan gems and avoid the red enemies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game. Collect all the gems to win, but be careful! "
        "Touching an enemy will cost you a life. The more gems you collect, the faster and more numerous the enemies become."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.WIN_CONDITION_GEMS = 20
        self.INITIAL_LIVES = 3
        self.WALL_THICKNESS = 10

        # --- Colors ---
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_INVINCIBLE = (255, 255, 180)
        self.COLOR_GEM = (0, 255, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_UI = (255, 255, 255)
        self.COLOR_HEART = (220, 20, 60)

        # --- Physics and Gameplay ---
        self.PLAYER_SPEED = 5.0
        self.PLAYER_SIZE = 16
        self.GEM_SIZE = 8
        self.ENEMY_SIZE = 20
        self.INITIAL_ENEMY_SPEED = 1.0
        self.ENEMY_SPEED_INCREMENT = 0.2
        self.ENEMY_SPEED_CAP = 3.0
        self.INITIAL_ENEMY_COUNT = 3
        self.MAX_ENEMY_COUNT = 10
        self.DIFFICULTY_INTERVAL = 5 # Add enemy/speed every 5 gems
        self.PLAYER_INVINCIBILITY_DURATION = 90 # 3 seconds at 30fps
        self.RISK_RADIUS = 80 # Radius for "near enemy" bonus

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Etc... State Variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.gems_collected_total = 0
        self.game_over = False
        self.game_won = False
        
        self.player_pos = None
        self.player_invincible_timer = 0
        
        self.gems = []
        self.enemies = []
        self.particles = []
        self.enemy_speed = self.INITIAL_ENEMY_SPEED
        
        self.np_random = None
        
    def _get_random_pos(self, margin=20):
        """Gets a random position within the playable area."""
        return pygame.Vector2(
            self.np_random.uniform(self.WALL_THICKNESS + margin, self.WIDTH - self.WALL_THICKNESS - margin),
            self.np_random.uniform(self.WALL_THICKNESS + margin, self.HEIGHT - self.WALL_THICKNESS - margin)
        )

    def _spawn_gem(self):
        """Spawns a new gem, avoiding the player."""
        while True:
            pos = self._get_random_pos(self.GEM_SIZE)
            if self.player_pos and pos.distance_to(self.player_pos) > self.PLAYER_SIZE + self.GEM_SIZE + 20:
                break
            elif not self.player_pos:
                break
        self.gems.append({'pos': pos, 'size': self.GEM_SIZE})

    def _spawn_enemy(self):
        """Spawns a new enemy, avoiding the player."""
        while True:
            pos = self._get_random_pos(self.ENEMY_SIZE)
            if self.player_pos and pos.distance_to(self.player_pos) > self.PLAYER_SIZE + self.ENEMY_SIZE + 50:
                break
            elif not self.player_pos:
                break
        target = self._get_random_pos(self.ENEMY_SIZE)
        self.enemies.append({'pos': pos, 'target': target, 'size': self.ENEMY_SIZE})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.gems_collected_total = 0
        self.game_over = False
        self.game_won = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_invincible_timer = 0
        
        self.enemy_speed = self.INITIAL_ENEMY_SPEED
        
        self.gems.clear()
        self.enemies.clear()
        self.particles.clear()
        
        for _ in range(5): # Start with 5 gems on screen
            self._spawn_gem()
            
        for _ in range(self.INITIAL_ENEMY_COUNT):
            self._spawn_enemy()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def _get_closest_entity(self, entities):
        """Helper to find the closest entity and its distance from a list."""
        if not entities:
            return None, float('inf')
        
        closest_dist = float('inf')
        closest_entity = None
        
        for entity in entities:
            dist = self.player_pos.distance_to(entity['pos'])
            if dist < closest_dist:
                closest_dist = dist
                closest_entity = entity
                
        return closest_entity, closest_dist
    
    def step(self, action):
        reward = self._calculate_reward(action)
        
        # --- Update Game State ---
        if not self.game_over:
            self.steps += 1
            if self.player_invincible_timer > 0:
                self.player_invincible_timer -= 1

            # Update enemies
            for enemy in self.enemies:
                if enemy['pos'].distance_to(enemy['target']) < self.enemy_speed:
                    enemy['target'] = self._get_random_pos(enemy['size'])
                
                direction = (enemy['target'] - enemy['pos']).normalize()
                enemy['pos'] += direction * self.enemy_speed

            # Update particles
            self.particles = [p for p in self.particles if p['life'] > 0]
            for p in self.particles:
                p['pos'] += p['vel']
                p['life'] -= 1
                p['size'] = max(0, p['size'] - 0.1)

            # --- Collision Detection & Events ---
            player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
            
            # Player-Gem collision
            for gem in self.gems[:]:
                gem_rect = pygame.Rect(gem['pos'].x - gem['size'], gem['pos'].y - gem['size'], gem['size']*2, gem['size']*2)
                if player_rect.colliderect(gem_rect):
                    self.gems.remove(gem)
                    self._spawn_gem()
                    self.gems_collected_total += 1
                    
                    self.score += 10
                    reward += 10 # Event reward
                    
                    # Risk bonus
                    _, dist_to_enemy = self._get_closest_entity(self.enemies)
                    if dist_to_enemy < self.RISK_RADIUS:
                        self.score += 5
                        reward += 5 # Bonus reward
                        self._create_particles(self.player_pos, 10, (255, 215, 0), 2) # Gold particles for bonus
                    
                    self._create_particles(gem['pos'], 20, self.COLOR_GEM) # Gem collection particles
                    
                    # Difficulty scaling
                    if self.gems_collected_total > 0 and self.gems_collected_total % self.DIFFICULTY_INTERVAL == 0:
                        if len(self.enemies) < self.MAX_ENEMY_COUNT:
                            self._spawn_enemy()
                        if self.enemy_speed < self.ENEMY_SPEED_CAP:
                            self.enemy_speed = min(self.ENEMY_SPEED_CAP, self.enemy_speed + self.ENEMY_SPEED_INCREMENT)

            # Player-Enemy collision
            if self.player_invincible_timer == 0:
                for enemy in self.enemies:
                    enemy_poly = self._get_triangle_points(enemy['pos'], enemy['size'], (enemy['target'] - enemy['pos']))
                    
                    # Check for collision between player rectangle and enemy polygon
                    collided = False
                    # 1. Check if any vertex of the enemy polygon is inside the player rect
                    if any(player_rect.collidepoint(p) for p in enemy_poly):
                        collided = True
                    
                    # 2. If not, check if any edge of the enemy polygon intersects the player rect.
                    if not collided:
                        for i in range(len(enemy_poly)):
                            p1 = enemy_poly[i]
                            p2 = enemy_poly[(i + 1) % len(enemy_poly)] # Wrap around for the last edge
                            if player_rect.clipline(p1, p2):
                                collided = True
                                break
                    
                    if collided:
                        self.lives -= 1
                        self.score = max(0, self.score - 20)
                        reward -= 20 # Event penalty
                        self.player_invincible_timer = self.PLAYER_INVINCIBILITY_DURATION
                        self._create_particles(self.player_pos, 30, self.COLOR_ENEMY, 3) # Player hit particles
                        break # Only one hit per frame
        
        terminated = self._check_termination()

        # Terminal rewards
        if terminated:
            if self.game_won:
                reward += 100 # Win bonus
            elif self.lives <= 0:
                reward -= 100 # Loss penalty
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _calculate_reward(self, action):
        """Calculates continuous rewards and applies player movement."""
        reward = 0
        if self.game_over:
            return 0
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Pre-move state for reward calculation ---
        _, dist_to_closest_gem_before = self._get_closest_entity(self.gems)
        _, dist_to_closest_enemy_before = self._get_closest_entity(self.enemies)

        # --- Handle player movement ---
        if movement == 1: # Up
            self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
        
        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.WALL_THICKNESS + self.PLAYER_SIZE/2, self.WIDTH - self.WALL_THICKNESS - self.PLAYER_SIZE/2)
        self.player_pos.y = np.clip(self.player_pos.y, self.WALL_THICKNESS + self.PLAYER_SIZE/2, self.HEIGHT - self.WALL_THICKNESS - self.PLAYER_SIZE/2)

        # --- Post-move state for reward calculation ---
        _, dist_to_closest_gem_after = self._get_closest_entity(self.gems)
        _, dist_to_closest_enemy_after = self._get_closest_entity(self.enemies)
        
        # Continuous rewards
        if dist_to_closest_gem_after < dist_to_closest_gem_before:
            reward += 0.1
        if dist_to_closest_enemy_after < dist_to_closest_enemy_before:
            reward -= 0.2
            
        return reward

    def _check_termination(self):
        """Checks and sets termination conditions."""
        if self.game_over:
            return True
            
        if self.gems_collected_total >= self.WIN_CONDITION_GEMS:
            self.game_over = True
            self.game_won = True
        elif self.lives <= 0:
            self.game_over = True
            self.game_won = False
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.game_won = False
        
        return self.game_over
    
    def _create_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

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
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.HEIGHT - self.WALL_THICKNESS, self.WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), color)
            except TypeError: # Handle potential color format issue
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), p['color'])

        # Draw gems
        for gem in self.gems:
            pos_int = (int(gem['pos'].x), int(gem['pos'].y))
            size_int = int(gem['size'])
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], size_int + 3, (*self.COLOR_GEM, 50))
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], size_int, self.COLOR_GEM)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], size_int, self.COLOR_GEM)
            
        # Draw enemies
        for enemy in self.enemies:
            direction = (enemy['target'] - enemy['pos'])
            points = self._get_triangle_points(enemy['pos'], enemy['size'], direction)
            # Glow effect
            pygame.gfxdraw.filled_polygon(self.screen, points, (*self.COLOR_ENEMY, 80))
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)

        # Draw player
        player_color = self.COLOR_PLAYER
        if self.player_invincible_timer > 0 and (self.steps // 3) % 2 == 0:
            player_color = self.COLOR_PLAYER_INVINCIBLE
        
        player_rect = pygame.Rect(
            int(self.player_pos.x - self.PLAYER_SIZE / 2),
            int(self.player_pos.y - self.PLAYER_SIZE / 2),
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        # Glow effect
        glow_rect = player_rect.inflate(6, 6)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*player_color, 80), glow_surf.get_rect(), border_radius=4)
        self.screen.blit(glow_surf, glow_rect.topleft)
        
        pygame.draw.rect(self.screen, player_color, player_rect, border_radius=3)

    def _get_triangle_points(self, pos, size, direction):
        if direction.length() > 0:
            direction = direction.normalize()
        else:
            direction = pygame.Vector2(0, -1)
        
        p1 = pos + direction * size * 0.8
        p2 = pos - direction * size * 0.4 + direction.rotate(90) * size * 0.6
        p3 = pos - direction * size * 0.4 + direction.rotate(-90) * size * 0.6
        return [(int(p.x), int(p.y)) for p in (p1, p2, p3)]

    def _draw_heart(self, surface, x, y, size):
        points = [
            (x, y + size * 0.25),
            (x - size * 0.5, y - size * 0.25),
            (x - size * 0.25, y - size * 0.6),
            (x, y - size * 0.25),
            (x + size * 0.25, y - size * 0.6),
            (x + size * 0.5, y - size * 0.25),
        ]
        pygame.gfxdraw.aapolygon(surface, points, self.COLOR_HEART)
        pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_HEART)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WALL_THICKNESS + 10, self.WALL_THICKNESS + 5))
        
        # Gems collected
        gem_text = self.font_ui.render(f"Gems: {self.gems_collected_total}/{self.WIN_CONDITION_GEMS}", True, self.COLOR_UI)
        gem_text_rect = gem_text.get_rect(centerx=self.WIDTH/2, y=self.WALL_THICKNESS + 5)
        self.screen.blit(gem_text, gem_text_rect)

        # Lives
        for i in range(self.lives):
            self._draw_heart(self.screen, self.WIDTH - self.WALL_THICKNESS - 20 - (i * 35), self.WALL_THICKNESS + 20, 20)
            
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            
            end_text = self.font_game_over.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "gems_collected": self.gems_collected_total,
        }
    
    def close(self):
        pygame.quit()

# Example of how to run the environment for human play
if __name__ == '__main__':
    # Allow Pygame to use a visible display if not in a headless environment
    if "SDL_VIDEODRIVER" in os.environ and os.environ['SDL_VIDEODRIVER'] == 'dummy':
        del os.environ['SDL_VIDEODRIVER']

    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    # Pygame display setup for human play
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Gem Collector")
        clock = pygame.time.Clock()
        human_play = True
    except pygame.error:
        print("Pygame display could not be initialized. Running in headless mode.")
        human_play = False

    if not human_play:
        # If display fails, just validate and exit
        env.close()
        exit()

    running = True
    terminated = False
    
    # Game loop for human play
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Map keys to MultiDiscrete action
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
                # Wait for a key press to reset
                pygame.time.wait(2000) # Show final screen for 2 seconds
                env.reset()
                terminated = False

        # Render the observation from the environment to the display
        frame = env._get_observation()
        # The observation is (H, W, C), but pygame blit needs a surface from (W, H, C)
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

    env.close()