
# Generated: 2025-08-27T20:24:12.216969
# Source Brief: brief_02444.md
# Brief Index: 2444

        
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
    An isometric arcade hack-and-slash game where the player uses a mighty hammer
    to defeat hordes of monsters. The game prioritizes visual flair and satisfying
    game feel, with smooth animations, particle effects, and responsive controls.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Press Space to swing your hammer. Survive the onslaught!"
    )

    game_description = (
        "Smash hordes of procedurally generated monsters in an isometric arcade game using a mighty hammer."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500 # 50 seconds
        self.WIN_CONDITION_KILLS = 25
        self.MAX_HEALTH = 5

        # World/Isometric constants
        self.WORLD_WIDTH, self.WORLD_HEIGHT = 24, 16
        self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF = 24, 12
        self.SCREEN_OFFSET_X = self.WIDTH // 2
        self.SCREEN_OFFSET_Y = 120

        # Player constants
        self.PLAYER_SPEED = 0.18
        self.PLAYER_INVINCIBILITY_FRAMES = 60 # 2 seconds

        # Hammer constants
        self.HAMMER_RADIUS = 2.5
        self.HAMMER_COOLDOWN = 20 # frames
        self.HAMMER_ANIMATION_DURATION = 8 # frames

        # Monster constants
        self.INITIAL_MONSTER_SPEED = 0.04
        self.MONSTER_SPAWN_RATE = 45 # frames
        self.MAX_MONSTERS = 15
        
        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_SHADOW = (0, 0, 0, 100)
        self.COLOR_MONSTER = (255, 80, 80)
        self.COLOR_MONSTER_HIT = (255, 255, 255)
        self.COLOR_HAMMER = (255, 220, 0)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_HEART = (255, 50, 50)
        self.COLOR_UI_HEART_EMPTY = (70, 70, 90)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        self.font_huge = pygame.font.Font(None, 80)
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.player_pos = None
        self.player_health = 0
        self.player_invincibility_timer = 0
        self.monsters = []
        self.particles = []
        self.monsters_defeated = 0
        self.monster_speed = 0.0
        self.monster_spawn_timer = 0
        self.hammer_cooldown = 0
        self.hammer_animation_timer = 0
        self.space_was_held = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([self.WORLD_WIDTH / 2.0, self.WORLD_HEIGHT / 2.0], dtype=np.float64)
        self.player_health = self.MAX_HEALTH
        self.player_invincibility_timer = 0
        
        self.monsters = []
        self.particles = []
        
        self.monsters_defeated = 0
        self.monster_speed = self.INITIAL_MONSTER_SPEED
        self.monster_spawn_timer = self.MONSTER_SPAWN_RATE
        
        self.hammer_cooldown = 0
        self.hammer_animation_timer = 0
        self.space_was_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Small penalty for every step to encourage efficiency

        self._update_timers()
        reward += self._handle_input_and_actions(action)
        reward += self._update_monsters()
        self._update_spawner()
        self._update_particles()

        self.steps += 1
        
        terminated = (self.player_health <= 0 or 
                      self.monsters_defeated >= self.WIN_CONDITION_KILLS or
                      self.steps >= self.MAX_STEPS)

        if terminated and not self.game_over:
            self.game_over = True
            if self.monsters_defeated >= self.WIN_CONDITION_KILLS:
                reward += 100.0  # Big reward for winning
            elif self.player_health <= 0:
                reward += -100.0 # Big penalty for losing
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_timers(self):
        if self.player_invincibility_timer > 0:
            self.player_invincibility_timer -= 1
        if self.hammer_cooldown > 0:
            self.hammer_cooldown -= 1
        if self.hammer_animation_timer > 0:
            self.hammer_animation_timer -= 1
        if self.monster_spawn_timer > 0:
            self.monster_spawn_timer -= 1
            
    def _handle_input_and_actions(self, action):
        movement, space_held, _ = action
        reward = 0

        # Player Movement
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] -= 1 # UP
        elif movement == 2: move_vec[1] += 1 # DOWN
        elif movement == 3: move_vec[0] -= 1 # LEFT
        elif movement == 4: move_vec[0] += 1 # RIGHT
        
        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec)
        
        self.player_pos += move_vec * self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WORLD_WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.WORLD_HEIGHT)

        # Hammer Swing
        if space_held and not self.space_was_held and self.hammer_cooldown == 0:
            # Sound: Play hammer swing SFX
            self.hammer_cooldown = self.HAMMER_COOLDOWN
            self.hammer_animation_timer = self.HAMMER_ANIMATION_DURATION
            hits = 0
            for monster in self.monsters:
                dist = np.linalg.norm(self.player_pos - np.array([monster['x'], monster['y']]))
                if dist <= self.HAMMER_RADIUS:
                    monster['health'] -= 1
                    monster['hit_timer'] = 10 # frames for white flash
                    self._create_particles(monster['x'], monster['y'], self.COLOR_MONSTER_HIT, 15, 2.0)
                    hits += 1
            
            if hits > 0:
                # Sound: Play monster hit SFX
                reward += hits * 1.0 # Reward for each monster hit
                if hits > 1:
                    reward += 2.0 # Bonus for multi-hit
                self.score += hits
                self.monsters_defeated += hits
                
                # Difficulty scaling
                if self.monsters_defeated // 5 > (self.monsters_defeated - hits) // 5:
                    self.monster_speed += 0.02

        self.space_was_held = bool(space_held)
        return reward

    def _update_monsters(self):
        reward = 0
        monsters_to_remove = []
        for i, monster in enumerate(self.monsters):
            # Health and removal
            if monster['health'] <= 0:
                monsters_to_remove.append(i)
                continue

            # Movement
            direction = self.player_pos - np.array([monster['x'], monster['y']])
            dist = np.linalg.norm(direction)
            if dist > 0.5: # Don't move if very close
                direction /= dist
                monster['x'] += direction[0] * self.monster_speed
                monster['y'] += direction[1] * self.monster_speed
            
            # Update hit flash timer
            if monster['hit_timer'] > 0:
                monster['hit_timer'] -= 1
            
            # Player collision
            if dist < 0.8 and self.player_invincibility_timer == 0:
                # Sound: Play player damage SFX
                self.player_health -= 1
                self.player_invincibility_timer = self.PLAYER_INVINCIBILITY_FRAMES
                reward -= 1.0 # Penalty for getting hit
                self._create_particles(self.player_pos[0], self.player_pos[1], self.COLOR_PLAYER, 30, 3.0)
        
        # Remove defeated monsters
        for i in sorted(monsters_to_remove, reverse=True):
            del self.monsters[i]
            
        return reward

    def _update_spawner(self):
        if self.monster_spawn_timer <= 0 and len(self.monsters) < self.MAX_MONSTERS:
            self.monster_spawn_timer = self.MONSTER_SPAWN_RATE
            
            # Spawn on a random edge
            edge = self.np_random.integers(0, 4)
            if edge == 0: # Top
                x, y = self.np_random.random() * self.WORLD_WIDTH, -1
            elif edge == 1: # Bottom
                x, y = self.np_random.random() * self.WORLD_WIDTH, self.WORLD_HEIGHT + 1
            elif edge == 2: # Left
                x, y = -1, self.np_random.random() * self.WORLD_HEIGHT
            else: # Right
                x, y = self.WORLD_WIDTH + 1, self.np_random.random() * self.WORLD_HEIGHT
            
            self.monsters.append({'x': x, 'y': y, 'health': 1, 'hit_timer': 0, 'bob': self.np_random.random() * math.pi * 2})

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _iso_to_screen(self, x, y):
        screen_x = int((x - y) * self.TILE_WIDTH_HALF + self.SCREEN_OFFSET_X)
        screen_y = int((x + y) * self.TILE_HEIGHT_HALF + self.SCREEN_OFFSET_Y)
        return screen_x, screen_y

    def _create_particles(self, x, y, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * max_speed
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed * 0.1,
                'vy': math.sin(angle) * speed * 0.1,
                'life': self.np_random.integers(10, 25),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_entities()
        self._render_hammer_swing()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for i in range(self.WORLD_WIDTH + 1):
            start = self._iso_to_screen(i, 0)
            end = self._iso_to_screen(i, self.WORLD_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for i in range(self.WORLD_HEIGHT + 1):
            start = self._iso_to_screen(0, i)
            end = self._iso_to_screen(self.WORLD_WIDTH, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

    def _render_entities(self):
        entities = []
        # Add player
        entities.append({'type': 'player', 'x': self.player_pos[0], 'y': self.player_pos[1]})
        # Add monsters
        for m in self.monsters:
            entities.append({'type': 'monster', 'x': m['x'], 'y': m['y'], 'data': m})
        
        # Sort by y-coordinate for correct isometric rendering
        entities.sort(key=lambda e: e['y'])
        
        for entity in entities:
            sx, sy = self._iso_to_screen(entity['x'], entity['y'])
            if entity['type'] == 'player':
                self._render_player(sx, sy)
            elif entity['type'] == 'monster':
                self._render_monster(sx, sy, entity['data'])

    def _render_player(self, sx, sy):
        # Invincibility flash
        if self.player_invincibility_timer > 0 and (self.steps // 3) % 2 == 0:
            return

        # Shadow
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, 12, self.COLOR_PLAYER_SHADOW)
        
        # Body with bobbing motion
        bob = math.sin(self.steps * 0.15) * 3
        player_rect = pygame.Rect(sx - 10, sy - 25 - bob, 20, 30)
        pygame.draw.ellipse(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.gfxdraw.aaellipse(self.screen, player_rect.centerx, player_rect.centery, player_rect.width//2, player_rect.height//2, self.COLOR_PLAYER)

    def _render_monster(self, sx, sy, monster):
        # Shadow
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, 8, self.COLOR_PLAYER_SHADOW)
        
        # Body with bobbing motion
        bob = math.sin(self.steps * 0.2 + monster['bob']) * 2
        color = self.COLOR_MONSTER_HIT if monster['hit_timer'] > 0 else self.COLOR_MONSTER
        points = [
            (sx, sy - 18 - bob),
            (sx - 8, sy - 10 - bob),
            (sx + 8, sy - 10 - bob)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_hammer_swing(self):
        if self.hammer_animation_timer > 0:
            progress = 1.0 - (self.hammer_animation_timer / self.HAMMER_ANIMATION_DURATION)
            radius_world = self.HAMMER_RADIUS * progress
            radius_screen = int(radius_world * self.TILE_WIDTH_HALF)
            
            px, py = self._iso_to_screen(self.player_pos[0], self.player_pos[1])
            
            alpha = int(200 * (1 - progress))
            color = (*self.COLOR_HAMMER, alpha)
            
            # Create a temporary surface for the transparent circle
            temp_surf = pygame.Surface((radius_screen*2, radius_screen*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius_screen, radius_screen, radius_screen, color)
            pygame.gfxdraw.aacircle(temp_surf, radius_screen, radius_screen, radius_screen, color)
            self.screen.blit(temp_surf, (px - radius_screen, py - radius_screen))

    def _render_particles(self):
        for p in self.particles:
            sx, sy = self._iso_to_screen(p['x'], p['y'])
            alpha = max(0, int(255 * (p['life'] / 25.0)))
            color = (*p['color'][:3], alpha)
            size = max(1, int(3 * (p['life'] / 25.0)))
            pygame.draw.circle(self.screen, color, (sx, sy), size)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Health
        for i in range(self.MAX_HEALTH):
            heart_x = self.WIDTH - 30 - (i * 35)
            heart_y = 30
            color = self.COLOR_UI_HEART if i < self.player_health else self.COLOR_UI_HEART_EMPTY
            pygame.gfxdraw.filled_circle(self.screen, heart_x, heart_y, 12, color)
            pygame.gfxdraw.aacircle(self.screen, heart_x, heart_y, 12, color)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.monsters_defeated >= self.WIN_CONDITION_KILLS else "GAME OVER"
            text = self.font_huge.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "monsters_defeated": self.monsters_defeated,
            "game_over": self.game_over
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """ Call this at the end of __init__ to verify implementation. """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Isometric Hammer Smash")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Human input mapping
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait for a moment before auto-resetting, or wait for 'R' key
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()