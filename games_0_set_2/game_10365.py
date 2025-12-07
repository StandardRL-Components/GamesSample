import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:16:19.892282
# Source Brief: brief_00365.md
# Brief Index: 365
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Fight off waves of evolving insects using a variety of elemental pesticides. "
        "Match the correct pesticide to the insect's weakness to survive."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to spray pesticide and shift to cycle between pesticide types."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_WALL = (50, 60, 80)
    COLOR_TEXT = (220, 220, 240)
    COLOR_PLAYER = (100, 255, 100)
    COLOR_PLAYER_GLOW = (100, 255, 100, 50)
    
    # Pesticide & Insect Type Mappings
    PESTICIDE_TYPES = ['WATER', 'OIL', 'FIRE']
    PESTICIDE_COLORS = {
        'WATER': (50, 150, 255),
        'OIL': (255, 200, 50),
        'FIRE': (255, 100, 50)
    }
    INSECT_DATA = {
        'DRIP_MITE': {'color': (80, 180, 255), 'weakness': 'WATER', 'health': 30, 'damage': 5, 'speed': 1.5},
        'GREASE_ROACH': {'color': (200, 160, 40), 'weakness': 'OIL', 'health': 50, 'damage': 8, 'speed': 2.0},
        'CINDER_CENTIPEDE': {'color': (255, 80, 30), 'weakness': 'FIRE', 'health': 70, 'damage': 12, 'speed': 1.2}
    }
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Initialize state variables (will be properly set in reset)
        self.steps = 0
        self.score = 0
        self.floor = 1
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_health = 100
        self.player_max_health = 100
        self.player_speed = 4.0
        self.player_size = 20
        self.insects = []
        self.particles = []
        self.damage_popups = []
        self.sprays = []
        self.current_pesticide_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.floor = 1
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.player_max_health
        
        self.insects = []
        self.particles = []
        self.damage_popups = []
        self.sprays = []
        
        self.current_pesticide_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._generate_floor()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        
        events = []
        
        # Store distance to closest insect before moving
        dist_before = self._get_closest_insect_dist()

        # Handle player actions
        self._handle_input(movement, space_pressed, shift_pressed, events)
        
        # Update game state
        self._update_game_state(events)

        # Store distance to closest insect after moving
        dist_after = self._get_closest_insect_dist()

        # Calculate reward
        reward = self._calculate_reward(events, dist_before, dist_after)
        self.score += reward

        # Check termination conditions
        self.steps += 1
        terminated = self._check_termination(events)
        if terminated:
            # Apply terminal reward
            if 'PLAYER_DIED' in events:
                reward -= 100
            elif 'FLOOR_CLEARED' in events:
                reward += 100
            self.game_over = True
        
        # Update previous action states
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed, events):
        # --- Movement ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y -= 1  # Up
        elif movement == 2: move_vec.y += 1  # Down
        elif movement == 3: move_vec.x -= 1  # Left
        elif movement == 4: move_vec.x += 1  # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.player_speed
        
        # Clamp player position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.player_size / 2, self.WIDTH - self.player_size / 2)
        self.player_pos.y = np.clip(self.player_pos.y, self.player_size / 2, self.HEIGHT - self.player_size / 2)

        # --- Cycle Pesticide ---
        if shift_pressed:
            self.current_pesticide_idx = (self.current_pesticide_idx + 1) % len(self.PESTICIDE_TYPES)
            # SFX: PESTICIDE_SWITCH

        # --- Attack ---
        if space_pressed:
            pesticide_type = self.PESTICIDE_TYPES[self.current_pesticide_idx]
            # Create a spray cone
            mouse_dir = self._get_aim_direction()
            for _ in range(30):
                angle = math.atan2(mouse_dir.y, mouse_dir.x) + random.uniform(-0.4, 0.4)
                speed = random.uniform(4, 7)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                pos = self.player_pos.copy()
                lifespan = random.randint(20, 40)
                self.sprays.append({'pos': pos, 'vel': vel, 'type': pesticide_type, 'lifespan': lifespan})
            # SFX: SPRAY

    def _update_game_state(self, events):
        player_rect = pygame.Rect(self.player_pos.x - self.player_size/2, self.player_pos.y - self.player_size/2, self.player_size, self.player_size)

        # --- Update Insects ---
        for insect in self.insects[:]:
            # AI: Move towards player
            direction = self.player_pos - insect['pos']
            if direction.length() > 0:
                direction.normalize_ip()
                insect['pos'] += direction * insect['data']['speed']
            
            # Animation
            insect['anim_offset'] += 0.2
            
            # Collision with player
            insect_rect = pygame.Rect(insect['pos'].x - insect['size']/2, insect['pos'].y - insect['size']/2, insect['size'], insect['size'])
            if player_rect.colliderect(insect_rect):
                if 'cooldown' not in insect or insect['cooldown'] == 0:
                    self.player_health -= insect['data']['damage']
                    events.append('PLAYER_HURT')
                    self._create_damage_popup(str(insect['data']['damage']), self.player_pos, (255, 50, 50))
                    insect['cooldown'] = self.FPS  # 1-second cooldown
                    # SFX: PLAYER_HURT
            if 'cooldown' in insect and insect['cooldown'] > 0:
                insect['cooldown'] -= 1

        # --- Update Sprays & Check Hits ---
        for spray in self.sprays[:]:
            spray['pos'] += spray['vel']
            spray['lifespan'] -= 1
            if spray['lifespan'] <= 0:
                self.sprays.remove(spray)
                continue

            spray_rect = pygame.Rect(spray['pos'].x - 2, spray['pos'].y - 2, 4, 4)
            for insect in self.insects[:]:
                insect_rect = pygame.Rect(insect['pos'].x - insect['size']/2, insect['pos'].y - insect['size']/2, insect['size'], insect['size'])
                if insect_rect.colliderect(spray_rect):
                    is_correct_type = (spray['type'] == insect['data']['weakness'])
                    damage = 10 if is_correct_type else 1
                    
                    insect['health'] -= damage
                    events.append('DAMAGE_DEALT')
                    if not is_correct_type:
                        events.append('WRONG_PESTICIDE')
                    
                    self._create_damage_popup(str(damage), insect['pos'], (255, 255, 255) if is_correct_type else (150, 150, 150))
                    self._create_hit_particles(insect['pos'], self.PESTICIDE_COLORS[spray['type']])
                    
                    if spray in self.sprays: self.sprays.remove(spray)
                    
                    if insect['health'] <= 0:
                        events.append('INSECT_KILLED')
                        if is_correct_type:
                            events.append('CORRECT_KILL')
                        self._create_death_particles(insect['pos'], insect['data']['color'])
                        self.insects.remove(insect)
                        # SFX: INSECT_DEATH
                    else:
                        # SFX: INSECT_HIT
                        pass
                    break

        # --- Update Particles & Popups ---
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0: self.particles.remove(p)

        for d in self.damage_popups[:]:
            d['pos'].y -= 0.5
            d['lifespan'] -= 1
            if d['lifespan'] <= 0: self.damage_popups.remove(d)

    def _generate_floor(self):
        self.insects.clear()
        num_insects = self.floor
        
        available_types = ['DRIP_MITE']
        if self.floor >= 2: available_types.append('GREASE_ROACH')
        if self.floor >= 4: available_types.append('CINDER_CENTIPEDE')
        
        for _ in range(num_insects):
            insect_type = random.choice(available_types)
            data = self.INSECT_DATA[insect_type]
            
            # Spawn away from the center
            while True:
                pos = pygame.Vector2(random.uniform(50, self.WIDTH-50), random.uniform(50, self.HEIGHT-50))
                if pos.distance_to(self.player_pos) > 150:
                    break

            self.insects.append({
                'pos': pos,
                'data': data,
                'health': data['health'],
                'max_health': data['health'],
                'size': 24 + data['health'] / 10,
                'anim_offset': random.uniform(0, 2 * math.pi)
            })

    def _calculate_reward(self, events, dist_before, dist_after):
        reward = 0
        
        # Movement reward
        if dist_before is not None and dist_after is not None:
             reward += (dist_before - dist_after) * 0.005 # Scaled down
        
        # Event-based rewards
        for event in events:
            if event == 'DAMAGE_DEALT': reward += 1.0
            if event == 'WRONG_PESTICIDE': reward -= 1.0
            if event == 'CORRECT_KILL': reward += 10.0
        
        return reward

    def _check_termination(self, events):
        if self.player_health <= 0:
            events.append('PLAYER_DIED')
            return True
        if not self.insects:
            events.append('FLOOR_CLEARED')
            # For this simple version, clearing the floor ends the episode.
            # A more complex game would call _generate_floor and increment self.floor.
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Walls ---
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), 10)

        # --- Draw Particles ---
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['size'], p['size']), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect())
            self.screen.blit(s, (int(p['pos'].x - p['size']/2), int(p['pos'].y - p['size']/2)))

        # --- Draw Sprays ---
        for spray in self.sprays:
            color = self.PESTICIDE_COLORS[spray['type']]
            pygame.draw.circle(self.screen, color, spray['pos'], 2)

        # --- Draw Insects ---
        for insect in self.insects:
            size = insect['size'] + math.sin(insect['anim_offset']) * 2
            pos = insect['pos']
            color = insect['data']['color']
            rect = pygame.Rect(int(pos.x - size/2), int(pos.y - size/2), int(size), int(size))
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in color), rect, 2)
            
            # Health bar
            bar_width = size
            bar_height = 5
            health_pct = insect['health'] / insect['max_health']
            pygame.draw.rect(self.screen, (50,0,0), (pos.x - bar_width/2, pos.y - size/2 - 10, bar_width, bar_height))
            pygame.draw.rect(self.screen, (255,0,0), (pos.x - bar_width/2, pos.y - size/2 - 10, bar_width * health_pct, bar_height))

        # --- Draw Player ---
        size = self.player_size + math.sin(self.steps * 0.2) * 2 # Shrinking/pulsating effect
        pos = self.player_pos
        
        # Glow
        glow_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (size, size), size)
        self.screen.blit(glow_surf, (int(pos.x-size), int(pos.y-size)))
        
        # Body
        rect = pygame.Rect(int(pos.x - size/2), int(pos.y - size/2), int(size), int(size))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect)
        
        # Aiming indicator
        aim_dir = self._get_aim_direction()
        aim_pos = pos + aim_dir * (size / 2 + 5)
        pygame.draw.circle(self.screen, self.PESTICIDE_COLORS[self.PESTICIDE_TYPES[self.current_pesticide_idx]], aim_pos, 3)


    def _render_ui(self):
        # --- Player Health Bar ---
        bar_width = 200
        bar_height = 20
        health_pct = self.player_health / self.player_max_health
        pygame.draw.rect(self.screen, (50,0,0), (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, (0,200,0), (10, 10, bar_width * health_pct, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (10, 10, bar_width, bar_height), 2)
        
        # --- Score & Floor ---
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        floor_text = self.font_small.render(f"Floor: {self.floor}", True, self.COLOR_TEXT)
        self.screen.blit(floor_text, (self.WIDTH - floor_text.get_width() - 10, 35))

        # --- Selected Pesticide ---
        p_type = self.PESTICIDE_TYPES[self.current_pesticide_idx]
        p_color = self.PESTICIDE_COLORS[p_type]
        p_text = self.font_small.render(f"Pesticide: {p_type}", True, p_color)
        self.screen.blit(p_text, (10, self.HEIGHT - 30))
        
        # --- Damage Popups ---
        for d in self.damage_popups:
            alpha = int(255 * (d['lifespan'] / d['max_lifespan']))
            d_text = self.font_small.render(d['text'], True, d['color'])
            d_text.set_alpha(alpha)
            self.screen.blit(d_text, d['pos'])

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "floor": self.floor}

    def close(self):
        pygame.quit()

    # --- Utility Methods ---
    def _get_aim_direction(self):
        # Aim towards the closest insect
        if not self.insects:
            return pygame.Vector2(1, 0)
        closest_insect = min(self.insects, key=lambda i: self.player_pos.distance_squared_to(i['pos']))
        direction = closest_insect['pos'] - self.player_pos
        if direction.length() > 0:
            return direction.normalize()
        return pygame.Vector2(1, 0)

    def _get_closest_insect_dist(self):
        if not self.insects:
            return None
        return min(self.player_pos.distance_to(i['pos']) for i in self.insects)

    def _create_damage_popup(self, text, pos, color):
        self.damage_popups.append({
            'text': text, 
            'pos': pos.copy(), 
            'color': color, 
            'lifespan': self.FPS // 2, 
            'max_lifespan': self.FPS // 2
        })

    def _create_hit_particles(self, pos, color):
        for _ in range(5):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = random.randint(10, 20)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'max_lifespan': lifespan, 'color': color, 'size': random.randint(2,4)})

    def _create_death_particles(self, pos, color):
        for _ in range(40):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = random.randint(20, 40)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'max_lifespan': lifespan, 'color': color, 'size': random.randint(3,6)})

    def validate_implementation(self):
        # This method is for self-checking and not part of the standard Gym API
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
    # This block allows you to play the game manually for testing
    # To run, you'll need to remove the "dummy" SDL_VIDEODRIVER
    # and install pygame.
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    total_reward = 0
    
    # Pygame setup for manual play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Pesticide Panic")
    clock = pygame.time.Clock()

    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)

    env.close()