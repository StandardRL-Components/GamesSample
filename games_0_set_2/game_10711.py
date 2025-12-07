import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper function for drawing anti-aliased text
def draw_text(surface, text, size, x, y, color, font_name=None, align="center"):
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if align == "center":
        text_rect.center = (x, y)
    elif align == "topleft":
        text_rect.topleft = (x, y)
    elif align == "topright":
        text_rect.topright = (x, y)
    elif align == "midbottom":
        text_rect.midbottom = (x, y)
    surface.blit(text_surface, text_rect)

# Helper function for drawing circles with a glow effect
def draw_glowing_circle(surface, x, y, radius, color, glow_color):
    # Draw glow layers
    for i in range(4):
        glow_radius = int(radius + i * 2)
        alpha = 80 - i * 20
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*glow_color, alpha), (glow_radius, glow_radius), glow_radius)
        surface.blit(glow_surf, (int(x - glow_radius), int(y - glow_radius)))
    
    # Draw main circle
    pygame.gfxdraw.filled_circle(surface, int(x), int(y), int(radius), color)
    pygame.gfxdraw.aacircle(surface, int(x), int(y), int(radius), color)

# Helper class for game entities
class Particle:
    def __init__(self, x, y, vx, vy, lifespan, color, p_type, data=None):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.lifespan = lifespan
        self.color = color
        self.type = p_type
        self.data = data or {}

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = "Control a bacterial colony, fending off rivals by launching word-based attacks and gathering nutrients to survive."
    user_guide = "Use ↑/↓ arrows to select an attack word. Press space to fire at the nearest enemy. Hold shift to gather nearby nutrients."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- CONSTANTS ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.MAX_ENEMIES = 10
        self.MAX_NUTRIENTS = 15

        # Colors
        self.COLOR_BG = (26, 26, 46)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 150)
        self.COLOR_ENEMY = (255, 0, 85)
        self.COLOR_ENEMY_GLOW = (150, 0, 50)
        self.COLOR_NUTRIENT = (61, 139, 61)
        self.COLOR_TEXT = (230, 230, 255)
        self.COLOR_HEALTH_BAR = (0, 200, 0)
        self.COLOR_HEALTH_BAR_BG = (80, 80, 80)
        
        # Word attacks
        self.ATTACK_WORDS = [
            {"word": "TOXIN", "damage": 15, "speed": 3, "color": (150, 255, 150)},
            {"word": "ACID", "damage": 25, "speed": 2, "color": (255, 255, 100)},
            {"word": "SPIKE", "damage": 10, "speed": 5, "color": (200, 200, 200)},
            {"word": "VIRUS", "damage": 20, "speed": 2.5, "color": (200, 100, 255)},
        ]
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables
        self.player_pos = np.array([0.0, 0.0])
        self.np_random = None
        self._init_state()

    def _init_state(self):
        """Initializes all game state variables. Called by reset()."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Player state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=float)
        self.player_health = 100
        self.player_max_health = 100
        self.player_resources = 50
        self.player_selected_word_idx = 0
        self.player_attack_cooldown = 0

        # Enemy state
        self.enemies = []
        self.enemy_spawn_timer = 150  # 5 seconds at 30 FPS
        self.enemy_spawn_rate = 150
        self.enemy_initial_spawn_count = 1

        # Nutrient state
        self.nutrients = []

        # Projectiles and effects
        self.particles = []

        # Action state tracking for edge detection
        self.prev_action = np.array([0, 0, 0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._init_state()

        # Populate initial game world
        for _ in range(self.enemy_initial_spawn_count):
            self._spawn_enemy()
        
        for _ in range(self.MAX_NUTRIENTS):
            self._spawn_nutrient()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        self.steps += 1
        
        # --- 1. Handle Player Actions ---
        movement, space_action, shift_action = action[0], action[1], action[2]
        
        # Detect rising edge for discrete actions
        up_pressed = movement == 1 and self.prev_action[0] != 1
        down_pressed = movement == 2 and self.prev_action[0] != 2
        space_pressed = space_action == 1 and self.prev_action[1] == 0
        shift_pressed = shift_action == 1 and self.prev_action[2] == 0
        
        # Action: Cycle through attack words
        if up_pressed:
            self.player_selected_word_idx = (self.player_selected_word_idx - 1) % len(self.ATTACK_WORDS)
        if down_pressed:
            self.player_selected_word_idx = (self.player_selected_word_idx + 1) % len(self.ATTACK_WORDS)

        # Action: Launch attack
        if space_pressed and self.player_attack_cooldown <= 0 and self.enemies:
            target = self._get_nearest_enemy(self.player_pos)
            if target:
                self.player_attack_cooldown = 20 # Cooldown of 2/3 second
                word_data = self.ATTACK_WORDS[self.player_selected_word_idx]
                self._create_word_projectile(self.player_pos, target, word_data, is_player=True)

        # Action: Gather resources
        if shift_pressed:
            nutrient = self._get_nearest_nutrient(self.player_pos)
            if nutrient:
                dist = np.linalg.norm(nutrient['pos'] - self.player_pos)
                if dist < 150: # Gathering range
                    self.player_resources += nutrient['value']
                    # FIX: list.remove() fails on dicts with numpy arrays due to ambiguous truth value.
                    # Use a list comprehension with an identity check ('is') to remove the object.
                    self.nutrients = [n for n in self.nutrients if n is not nutrient]
                    self._spawn_nutrient()
                    reward += 0.1
                    # Create gathering particle effect
                    for _ in range(10):
                        p_start = nutrient['pos']
                        p_vel = (self.player_pos - p_start) / 30.0 + self.np_random.uniform(-0.5, 0.5, 2)
                        self.particles.append(Particle(p_start[0], p_start[1], p_vel[0], p_vel[1], 30, self.COLOR_NUTRIENT, 'gather'))

        self.prev_action = action

        # --- 2. Update Game Logic ---
        # Update cooldowns
        if self.player_attack_cooldown > 0: self.player_attack_cooldown -= 1

        # Update enemies
        for enemy in self.enemies:
            enemy['attack_cooldown'] -= 1
            if enemy['attack_cooldown'] <= 0:
                enemy['attack_cooldown'] = self.np_random.integers(120, 200) # 4-6.6s cooldown
                word_data = self.ATTACK_WORDS[self.np_random.choice(len(self.ATTACK_WORDS))]
                self._create_word_projectile(enemy['pos'], None, word_data, is_player=False)

        # Update particles (projectiles, effects)
        reward += self._update_particles()
        
        # Spawn new enemies
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0 and len(self.enemies) < self.MAX_ENEMIES:
            self._spawn_enemy()
            self.enemy_spawn_timer = self.enemy_spawn_rate
        
        # Increase difficulty
        if self.steps > 0 and self.steps % 200 == 0:
            self.enemy_spawn_rate = max(60, self.enemy_spawn_rate * 0.99) # Capped at 2s spawn rate

        # --- 3. Check Termination Conditions ---
        terminated = False
        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            self.win = False
            reward -= 100
        
        if len(self.enemies) == 0 and self.steps > 1: # Win condition
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100

        truncated = False
        if self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _update_particles(self):
        step_reward = 0.0
        for p in self.particles[:]:
            p.lifespan -= 1
            if p.lifespan <= 0:
                if p in self.particles: self.particles.remove(p)
                continue
            
            p.x += p.vx
            p.y += p.vy
            
            pos = np.array([p.x, p.y])
            if p.type == 'word_projectile':
                removed = False
                if p.data['is_player']:
                    for enemy in self.enemies[:]:
                        dist = np.linalg.norm(pos - enemy['pos'])
                        if dist < enemy['radius']:
                            step_reward += self._damage_enemy(enemy, p.data['damage'])
                            self._create_impact_effect(pos, p.color)
                            if p in self.particles: self.particles.remove(p)
                            removed = True
                            break
                else: # Enemy projectile
                    dist = np.linalg.norm(pos - self.player_pos)
                    if dist < 20: # Player radius
                        self.player_health = max(0, self.player_health - p.data['damage'])
                        step_reward -= 0.1
                        self._create_impact_effect(pos, p.color)
                        if p in self.particles: self.particles.remove(p)
                        removed = True
                if removed:
                    continue

        return step_reward

    def _damage_enemy(self, enemy, damage):
        reward = 1.0
        enemy['health'] -= damage
        if enemy['health'] <= 0:
            reward += 5.0
            self._create_impact_effect(enemy['pos'], self.COLOR_ENEMY, 20, 2.0)
            # FIX: list.remove() fails on dicts with numpy arrays due to ambiguous truth value.
            # Use a list comprehension with an identity check ('is') to remove the object.
            self.enemies = [e for e in self.enemies if e is not enemy]
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw nutrients
        for n in self.nutrients:
            pygame.gfxdraw.filled_circle(self.screen, int(n['pos'][0]), int(n['pos'][1]), n['radius'], self.COLOR_NUTRIENT)
            pygame.gfxdraw.aacircle(self.screen, int(n['pos'][0]), int(n['pos'][1]), n['radius'], self.COLOR_NUTRIENT)

        # Draw player colony
        player_radius = 20 + 2 * math.sin(self.steps * 0.1)
        draw_glowing_circle(self.screen, self.player_pos[0], self.player_pos[1], player_radius, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
        
        # Draw enemy colonies
        for enemy in self.enemies:
            enemy['radius'] = 15 + 2 * math.sin(self.steps * 0.15 + enemy['pos'][0])
            draw_glowing_circle(self.screen, enemy['pos'][0], enemy['pos'][1], enemy['radius'], self.COLOR_ENEMY, self.COLOR_ENEMY_GLOW)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p.lifespan / p.data.get('max_life', p.lifespan+1)))
            color = (*p.color, alpha) if len(p.color) == 3 else p.color
            
            temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)

            if p.type == 'word_projectile':
                draw_text(temp_surf, p.data['word'], 20, p.x, p.y, color, align="center")
            elif p.type == 'impact':
                pygame.draw.circle(temp_surf, color, (int(p.x), int(p.y)), int(p.data['radius']))
            elif p.type == 'gather':
                 pygame.draw.circle(temp_surf, p.color, (int(p.x), int(p.y)), 2)
            
            temp_surf.set_alpha(alpha)
            self.screen.blit(temp_surf, (0,0))


    def _render_ui(self):
        # Health bars
        self._draw_health_bar(self.player_pos[0], self.player_pos[1] - 35, self.player_health, self.player_max_health)
        for enemy in self.enemies:
            self._draw_health_bar(enemy['pos'][0], enemy['pos'][1] - 25, enemy['health'], enemy['max_health'])

        # Player UI
        draw_text(self.screen, f"RESOURCES: {self.player_resources}", 22, 10, 10, self.COLOR_TEXT, align="topleft")
        
        # Selected word display
        word_data = self.ATTACK_WORDS[self.player_selected_word_idx]
        draw_text(self.screen, "ATTACK:", 20, self.player_pos[0], self.HEIGHT - 45, self.COLOR_TEXT, align="center")
        draw_text(self.screen, word_data['word'], 28, self.player_pos[0], self.HEIGHT - 20, word_data['color'], align="center")

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU ARE VICTORIOUS" if self.win else "COLONY DESTROYED"
            color = self.COLOR_PLAYER if self.win else self.COLOR_ENEMY
            draw_text(self.screen, msg, 50, self.WIDTH/2, self.HEIGHT/2 - 20, color, align="center")
            draw_text(self.screen, f"Final Score: {self.score:.1f}", 30, self.WIDTH/2, self.HEIGHT/2 + 30, self.COLOR_TEXT, align="center")

    def _draw_health_bar(self, x, y, current, maximum, width=50, height=7):
        if current < 0: current = 0
        fill_ratio = current / maximum
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (x - width/2, y - height/2, width, height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (x - width/2, y - height/2, width * fill_ratio, height))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "enemies_left": len(self.enemies),
            "resources": self.player_resources
        }

    # --- Spawning and Entity Management ---
    def _spawn_enemy(self):
        if len(self.enemies) >= self.MAX_ENEMIES: return
        edge = self.np_random.integers(4)
        if edge == 0: # top
            pos = np.array([self.np_random.uniform(0, self.WIDTH), -30.0])
        elif edge == 1: # bottom
            pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 30.0])
        elif edge == 2: # left
            pos = np.array([-30.0, self.np_random.uniform(0, self.HEIGHT)])
        else: # right
            pos = np.array([self.WIDTH + 30.0, self.np_random.uniform(0, self.HEIGHT)])
        
        self.enemies.append({
            'pos': pos,
            'health': 50, 'max_health': 50,
            'radius': 15,
            'attack_cooldown': self.np_random.integers(60, 120)
        })

    def _spawn_nutrient(self):
        if len(self.nutrients) >= self.MAX_NUTRIENTS: return
        self.nutrients.append({
            'pos': self.np_random.uniform([50, 50], [self.WIDTH - 50, self.HEIGHT - 50]),
            'radius': self.np_random.integers(5, 10),
            'value': 10
        })

    def _create_word_projectile(self, start_pos, target_enemy, word_data, is_player):
        if is_player:
            target_pos = target_enemy['pos']
        else: # Enemy targets player
            target_pos = self.player_pos + self.np_random.uniform(-20, 20, 2)
        
        direction = target_pos - start_pos
        dist = np.linalg.norm(direction)
        if dist == 0: return # Avoid division by zero
        
        direction = direction / dist
        velocity = direction * word_data['speed']
        
        lifespan = int(dist / word_data['speed']) + 10
        
        p_data = word_data.copy()
        p_data['is_player'] = is_player
        p_data['max_life'] = lifespan
        
        self.particles.append(Particle(
            start_pos[0], start_pos[1], velocity[0], velocity[1], lifespan,
            word_data['color'], 'word_projectile', p_data
        ))
    
    def _create_impact_effect(self, pos, color, num_particles=15, speed=1.5):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed * self.np_random.uniform(0.5, 1.5)
            lifespan = self.np_random.integers(10, 20)
            p_data = {'radius': self.np_random.uniform(1, 4), 'max_life': lifespan}
            self.particles.append(Particle(pos[0], pos[1], vel[0], vel[1], lifespan, color, 'impact', p_data))

    def _get_nearest_enemy(self, pos):
        if not self.enemies: return None
        return min(self.enemies, key=lambda e: np.linalg.norm(e['pos'] - pos))

    def _get_nearest_nutrient(self, pos):
        if not self.nutrients: return None
        return min(self.nutrients, key=lambda n: np.linalg.norm(n['pos'] - pos))

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play ---
    obs, info = env.reset(seed=42)
    done = False
    
    # Create a real display for manual play
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bacterial Colony Word Combat")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = np.array([0, 0, 0]) # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset(seed=42)
                total_reward = 0
                done = False

        if not done:
            keys = pygame.key.get_pressed()
            
            # This is a rising-edge detection system for manual play,
            # but the environment itself handles this from the action array.
            # For simplicity, we'll just map held keys to actions.
            # The env will correctly interpret them based on prev_action state.
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering ---
        # The environment returns the rendered frame as the observation.
        # We just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
    env.close()